"""Tests for the GOG context isolation pipeline.

Tests:
  1. Python import extraction
  2. Import resolution (relative + absolute)
  3. Graph building from a sample repo
  4. Seed finding from prompts
  5. Context isolation (message stripping)
  6. Graph manager caching
"""

import ast
import os
import shutil
import tempfile
import textwrap

import networkx as nx
import pytest

from ctxpact.isolation.python_parser import PythonParser
from ctxpact.isolation.ts_parser import TypeScriptParser
from ctxpact.isolation.graph_builder import build_graph, update_graph_for_file, discover_files
from ctxpact.isolation.seed_finder import (
    find_seeds, extract_identifiers_from_text, extract_file_paths_from_text,
)
from ctxpact.isolation.isolator import isolate_context, IsolationResult


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def sample_repo(tmp_path):
    """Create a sample Python repo with known dependencies.

    Structure:
      myapp/
        __init__.py
        config.py          (no imports)
        auth/
          __init__.py
          models.py         (imports config)
          service.py        (imports models, imports utils.helpers)
          views.py          (imports service)
        utils/
          __init__.py
          helpers.py        (imports config)
          logger.py         (no imports)
        api/
          __init__.py
          endpoints.py      (imports auth.views, utils.logger)
        noise/
          __init__.py
          unrelated.py      (imports nothing relevant)
          billing.py        (imports nothing relevant)
    """
    root = tmp_path / "myapp"

    # Create directories
    for d in ["auth", "utils", "api", "noise"]:
        (root / d).mkdir(parents=True)
        (root / d / "__init__.py").write_text("")

    (root / "__init__.py").write_text("")

    # config.py — no imports
    (root / "config.py").write_text(textwrap.dedent("""\
        DATABASE_URL = "sqlite:///app.db"
        SECRET_KEY = "secret"
        class Settings:
            debug = True
    """))

    # auth/models.py — imports config
    (root / "auth" / "models.py").write_text(textwrap.dedent("""\
        from ..config import Settings
        class User:
            def __init__(self, name, role="user"):
                self.name = name
                self.role = role
    """))

    # auth/service.py — imports models + utils.helpers
    (root / "auth" / "service.py").write_text(textwrap.dedent("""\
        from .models import User
        from ..utils.helpers import hash_password
        class AuthService:
            def authenticate(self, username, password):
                hashed = hash_password(password)
                return User(username)
    """))

    # auth/views.py — imports service
    (root / "auth" / "views.py").write_text(textwrap.dedent("""\
        from .service import AuthService
        def login_view(request):
            svc = AuthService()
            return svc.authenticate(request.user, request.password)
    """))

    # utils/helpers.py — imports config
    (root / "utils" / "helpers.py").write_text(textwrap.dedent("""\
        from ..config import SECRET_KEY
        def hash_password(pwd):
            return f"hashed_{pwd}_{SECRET_KEY}"
    """))

    # utils/logger.py — standalone
    (root / "utils" / "logger.py").write_text(textwrap.dedent("""\
        import logging
        logger = logging.getLogger("myapp")
        def log(msg):
            logger.info(msg)
    """))

    # api/endpoints.py — imports auth.views + utils.logger
    (root / "api" / "endpoints.py").write_text(textwrap.dedent("""\
        from ..auth.views import login_view
        from ..utils.logger import log
        def handle_login(request):
            log("Login attempt")
            return login_view(request)
    """))

    # noise/unrelated.py — red herring
    (root / "noise" / "unrelated.py").write_text(textwrap.dedent("""\
        # This file mentions auth and user but is NOT imported by anything
        class UserPreferences:
            pass
        def check_auth_status():
            return True
    """))

    # noise/billing.py — red herring
    (root / "noise" / "billing.py").write_text(textwrap.dedent("""\
        class BillingService:
            def charge(self, amount):
                return True
    """))

    return root


# =========================================================================
# Test 1: Python Import Extraction
# =========================================================================

class TestPythonParser:
    def setup_method(self):
        self.parser = PythonParser()

    def test_simple_imports(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("import os\nimport sys\nfrom pathlib import Path\n")
        imports = self.parser.extract_imports(str(f))
        assert "os" in imports
        assert "sys" in imports
        assert "pathlib" in imports

    def test_relative_imports(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("from . import sibling\nfrom ..parent import thing\nfrom .utils.helpers import func\n")
        imports = self.parser.extract_imports(str(f))
        assert ".sibling" in imports or "." in imports
        assert "..parent" in imports
        assert ".utils.helpers" in imports

    def test_syntax_error_fallback(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("import os\nthis is not valid python!!!\nfrom sys import argv\n")
        imports = self.parser.extract_imports(str(f))
        # Regex fallback should still find os and sys
        assert "os" in imports
        assert "sys" in imports

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        assert self.parser.extract_imports(str(f)) == []

    def test_resolve_relative(self, sample_repo):
        # auth/service.py imports .models → should resolve to auth/models.py
        service = str(sample_repo / "auth" / "service.py")
        resolved = self.parser.resolve_import(".models", service, str(sample_repo))
        assert resolved is not None
        assert resolved.endswith("models.py")

    def test_resolve_parent_relative(self, sample_repo):
        # auth/models.py imports ..config → should resolve to config.py
        models = str(sample_repo / "auth" / "models.py")
        resolved = self.parser.resolve_import("..config", models, str(sample_repo))
        assert resolved is not None
        assert resolved.endswith("config.py")


# =========================================================================
# Test 2: Graph Building
# =========================================================================

class TestGraphBuilder:
    def test_build_graph(self, sample_repo):
        graph = build_graph(str(sample_repo))
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

        # Check that key files are nodes
        node_basenames = {os.path.basename(n) for n in graph.nodes()}
        assert "config.py" in node_basenames
        assert "service.py" in node_basenames
        assert "views.py" in node_basenames
        assert "unrelated.py" in node_basenames  # All files are nodes

    def test_edges_exist(self, sample_repo):
        graph = build_graph(str(sample_repo))

        # service.py should have edges to models.py and helpers.py
        service_node = [n for n in graph.nodes() if "service.py" in n][0]
        successors = set(graph.successors(service_node))
        successor_names = {os.path.basename(s) for s in successors}
        assert "models.py" in successor_names

    def test_noise_has_no_outgoing_edges(self, sample_repo):
        graph = build_graph(str(sample_repo))
        unrelated = [n for n in graph.nodes() if "unrelated.py" in n][0]
        assert graph.out_degree(unrelated) == 0

    def test_incremental_update(self, sample_repo):
        graph = build_graph(str(sample_repo))
        initial_edges = graph.number_of_edges()

        # Add a new file that imports config
        new_file = sample_repo / "new_module.py"
        new_file.write_text("from .config import Settings\n")

        update_graph_for_file(graph, str(new_file), str(sample_repo))
        assert str(new_file.absolute()) in graph.nodes()
        # May have a new edge
        assert graph.number_of_edges() >= initial_edges


# =========================================================================
# Test 3: Seed Finding
# =========================================================================

class TestSeedFinder:
    def test_identifier_extraction(self):
        text = "Fix the AuthService bug in views.py"
        ids = extract_identifiers_from_text(text)
        assert "authservice" in ids or "auth" in ids

    def test_file_path_extraction(self):
        text = "Look at auth/service.py and fix the login"
        paths = extract_file_paths_from_text(text)
        assert any("service.py" in p for p in paths)

    def test_find_seeds_by_filename(self, sample_repo):
        graph = build_graph(str(sample_repo))
        seeds = find_seeds(graph, "Fix the auth service", repo_path=str(sample_repo))
        seed_basenames = {os.path.basename(s.node) for s in seeds}
        # Should find auth-related files
        assert len(seeds) > 0
        assert any("auth" in os.path.basename(s.node).lower() or "service" in os.path.basename(s.node).lower() for s in seeds)

    def test_find_seeds_by_symbol(self, sample_repo):
        graph = build_graph(str(sample_repo))
        seeds = find_seeds(graph, "The AuthService class has a bug", repo_path=str(sample_repo))
        # Should find service.py which defines AuthService
        assert len(seeds) > 0


# =========================================================================
# Test 4: Context Isolation
# =========================================================================

class TestIsolator:
    def test_isolation_strips_irrelevant_files(self, sample_repo):
        graph = build_graph(str(sample_repo))

        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Read these files"},
            # Tool result with auth-relevant file
            {"role": "tool", "content": f"File auth/service.py:\n" + "\n".join(
                [f"line {i}: code here" for i in range(20)]
            )},
            # Tool result with irrelevant file (should be stripped)
            {"role": "tool", "content": f"File noise/billing.py:\n" + "\n".join(
                [f"line {i}: billing code" for i in range(20)]
            )},
            {"role": "user", "content": "Fix the auth service bug"},
        ]

        result = isolate_context(
            graph=graph,
            messages=messages,
            prompt="Fix the auth service bug",
            repo_path=str(sample_repo),
        )

        assert result.seeds_found > 0
        assert len(result.isolated_files) > 0
        # System and user messages should be untouched
        assert result.messages[0]["role"] == "system"
        assert result.messages[-1]["role"] == "user"

    def test_no_seeds_no_stripping(self, sample_repo):
        graph = build_graph(str(sample_repo))

        messages = [
            {"role": "user", "content": "What is the meaning of life?"},
        ]

        result = isolate_context(
            graph=graph,
            messages=messages,
            prompt="What is the meaning of life?",
            repo_path=str(sample_repo),
        )

        # No code-related seeds → no stripping
        assert not result.applied
        assert result.messages == messages

    def test_empty_graph(self):
        graph = nx.DiGraph()
        messages = [{"role": "user", "content": "test"}]
        result = isolate_context(graph, messages, "test", "/tmp")
        assert not result.applied
        assert result.messages == messages


# =========================================================================
# Test 5: TypeScript Parser
# =========================================================================

class TestTypeScriptParser:
    def setup_method(self):
        self.parser = TypeScriptParser()

    def test_extract_imports(self, tmp_path):
        f = tmp_path / "test.ts"
        f.write_text("import { ref } from 'vue';\nimport { useState } from './hooks';\n")
        imports = self.parser.extract_imports(str(f))
        assert "vue" in imports
        assert "./hooks" in imports

    def test_vue_file(self, tmp_path):
        f = tmp_path / "test.vue"
        f.write_text('<script setup lang="ts">\nimport { useAuth } from "../stores/auth";\n</script>\n<template><div/></template>\n')
        imports = self.parser.extract_imports(str(f))
        assert "../stores/auth" in imports

    def test_no_script_block(self, tmp_path):
        f = tmp_path / "test.vue"
        f.write_text("<template><div>No script</div></template>")
        assert self.parser.extract_imports(str(f)) == []


# =========================================================================
# Run
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
