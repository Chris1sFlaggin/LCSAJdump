from setuptools import setup, find_packages
import subprocess
import sys
import os

def ask_install_plugins():
    """Interactively ask user if they want to install plugins during pip install."""
    print("\n" + "="*60)
    print("LCSAJdump v2.0.5")
    print("="*60)
    print("\nLCSAJdump includes optional integrations for popular tools:")
    print("  1. GDB/pwndbg Plugin  - Run 'lcsaj' command inside the debugger")
    print("  2. pwntools Helper    - Python API: LCSAJGadgets()")
    print("  3. IDA Pro Plugin     - GUI panel for gadget analysis")
    print("\nThese plugins are optional - the core tool works without them.")
    print("-"*60)

    plugins_to_install = []

    # Ask about each plugin
    try:
        # GDB Plugin
        response = input("\nInstall GDB/pwndbg plugin? [Y/n]: ").strip().lower()
        if response in ('', 'y', 'yes'):
            plugins_to_install.append('--gdb')

        # Pwntools Helper
        response = input("Install pwntools helper? [Y/n]: ").strip().lower()
        if response in ('', 'y', 'yes'):
            plugins_to_install.append('--pwntools')

        # IDA Plugin (always ask, even though it's WIP)
        response = input("Install IDA Pro plugin? [y/N]: ").strip().lower()
        if response in ('y', 'yes'):
            plugins_to_install.append('--ida')

    except (EOFError, KeyboardInterrupt):
        # Non-interactive mode (e.g., CI/CD) - install nothing
        print("\nNon-interactive mode detected. Skipping plugin installation.")
        return []

    return plugins_to_install

def run_install_script(plugins):
    """Run the install_integrations.sh script with selected plugins."""
    if not plugins:
        return

    script_path = os.path.join(os.path.dirname(__file__), 'install_integrations.sh')

    if os.path.exists(script_path):
        print(f"\n[*] Running plugin installer with: {' '.join(plugins)}")
        try:
            result = subprocess.run(
                ['bash', script_path] + plugins,
                check=True,
                capture_output=False,
                text=True
            )
            print("\n[+] Plugin installation complete!")
        except subprocess.CalledProcessError as e:
            print(f"\n[!] Plugin installation encountered issues: {e}", file=sys.stderr)
            print("[!] You can install plugins manually later with: ./install_integrations.sh")
    else:
        print(f"\n[!] Install script not found at {script_path}")
        print("[!] You can install plugins manually later with: ./install_integrations.sh")

# Check if we're in install mode (not just building)
if 'install' in sys.argv or 'develop' in sys.argv or '-e' in sys.argv:
    # Delay import to avoid issues during egg_info
    try:
        plugins = ask_install_plugins()
        if plugins:
            run_install_script(plugins)
        else:
            print("\n[*] No plugins selected. Core LCSAJdump installed successfully.")
            print("[*] Install plugins later with: ./install_integrations.sh")
    except Exception as e:
        print(f"\n[!] Plugin installation skipped due to error: {e}", file=sys.stderr)
        print("[*] Core LCSAJdump installed. Install plugins manually with: ./install_integrations.sh")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lcsajdump",
    version="2.0.5",
    author="Chris1sFlaggin",
    author_email="lcsajdump@chris1sflaggin.it",
    description="A Graph-Based ROP Gadget Finder for every architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://chris1sflaggin.it/LCSAJdump/",

    packages=find_packages(exclude=[
        "testCTFs*",
        "unitTest*",
        "_images*",
        "build*",
        "dist*",
        "venv*",
        "lcsajdump.venv*",
        "ml_study*",
        ".pytest_cache*"
    ]),

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Security",
        "Environment :: Console",
    ],
    python_requires='>=3.6',
    install_requires=[
        "capstone",
        "pyelftools",
        "click",
        "regex",
    ],
    extras_require={
        'ml': ['lightgbm', 'pandas', 'numpy', 'shap'],
        'full': ['lightgbm', 'pandas', 'numpy', 'shap', 'angr'],
    },
    entry_points={
        'console_scripts': [
            'lcsajdump=lcsajdump.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'lcsajdump': ['ml/models/*.pkl'],
    },
)
