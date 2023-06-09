# Set the name of the bazel workspace.
workspace(name = "langchain_workspace")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Download the rules_go and bazel_gazelle ruleset for augo-generating BUILD files.
http_archive(
    name = "io_bazel_rules_go",
    sha256 = "6b65cb7917b4d1709f9410ffe00ecf3e160edf674b78c54a894471320862184f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.39.0/rules_go-v0.39.0.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.39.0/rules_go-v0.39.0.zip",
    ],
)

http_archive(
    name = "bazel_gazelle",
    sha256 = "ecba0f04f96b4960a5b250c8e8eeec42281035970aa8852dda73098274d14a1d",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.29.0/bazel-gazelle-v0.29.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.29.0/bazel-gazelle-v0.29.0.tar.gz",
    ],
)

# Load rules_go ruleset and expose the toolchain and dep rules.
load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")
load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

# go_rules_dependencies is a function that registers external dependencies needed by the Go rules.
# See: https://github.com/bazelbuild/rules_go/blob/master/go/dependencies.rst#go_rules_dependencies
go_rules_dependencies()

# go_rules_dependencies is a function that registers external dependencies needed by the Go rules.
# See: https://github.com/bazelbuild/rules_go/blob/master/go/dependencies.rst#go_rules_dependencies
go_register_toolchains(version = "1.19.4")

# The following call configured the gazelle dependencies, Go environment and Go SDK.
gazelle_dependencies()

# Set up rules_python.
http_archive(
    name = "rules_python",
    sha256 = "94750828b18044533e98a129003b6a68001204038dc4749f40b195b24c38f49f",
    strip_prefix = "rules_python-0.21.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.21.0/rules_python-0.21.0.tar.gz",
)

http_archive(
    name = "rules_python_gazelle_plugin",
    sha256 = "94750828b18044533e98a129003b6a68001204038dc4749f40b195b24c38f49f",
    strip_prefix = "rules_python-0.21.0/gazelle",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.21.0/rules_python-0.21.0.tar.gz",
)

# Next we load the toolchain from rules_python.
load("@rules_python//python:repositories.bzl", "python_register_toolchains")

# We now register a hermetic Python interpreter rather than relying on a system-installed interpreter.
# This toolchain will allow bazel to download a specific python version, and use that version
# for compilation.
python_register_toolchains(
    name = "python310",
    python_version = "3.10",
)

# Load the interpreter and pip_parse rules.
load("@python310//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")

# This macro wraps the `pip_repository` rule that invokes `pip`, with `incremental` set.
# Accepts a locked/compiled requirements file and installs the dependencies listed within.
# Those dependencies become available in a generated `requirements.bzl` file.
# You can instead check this `requirements.bzl` file into your repo.
pip_parse(
    name = "pip",
    # Generate user friendly alias labels for each dependency that we have.
    incompatible_generate_aliases = True,
    # (Optional) You can provide a python_interpreter (path) or a python_interpreter_target (a Bazel target, that
    # acts as an executable). The latter can be anything that could be used as Python interpreter. E.g.:
    # 1. Python interpreter that you compile in the build file.
    # 2. Pre-compiled python interpreter included with http_archive.
    # 3. Wrapper script, like in the autodetecting python toolchain.
    #
    # Here, we use the interpreter constant that resolves to the host interpreter from the default Python toolchain.
    python_interpreter_target = interpreter,
    # Set the location of the lock file.
    requirements_lock = "//:requirements_lock.txt",
)

# Load the install_deps macro.
load("@pip//:requirements.bzl", "install_deps")

# Initialize repositories for all packages in requirements_lock.txt.
install_deps()

# The rules_python gazelle extension has some third-party go dependencies
# which we need to fetch in order to compile it.
load("@rules_python_gazelle_plugin//:deps.bzl", _py_gazelle_deps = "gazelle_deps")

# See: https://github.com/bazelbuild/rules_python/blob/main/gazelle/README.md
# This rule loads and compiles various go dependencies that running gazelle
# for python requirements.
_py_gazelle_deps()
