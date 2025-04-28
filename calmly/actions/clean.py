import os
import shutil
import argparse

def clean_project(
    yes: bool = False,
    quiet: bool = False,
    only_print_targets: bool = False
):
    """
    Remove generated subdirectories like models, logs, data, and evaluation.
    
    Parameters
    ----------
    yes : bool
        If True, do not ask for confirmation.
    quiet : bool
        If True, do not print anything to stdout. This also overrides the 'yes' parameter to yes=True
    only_print_targets: bool
        If True, only lists the targets for deletion. No actual deletion is performed. Parameters 'quiet' and 'yes' will be ignored.
    """
    targets = ["models", "logs", "data", "evaluation"]
    if only_print_targets:
        print
    
    if quiet and (not only_print_targets):
        yes = True
    else:
        print("The following directories will be removed (if they exist):")
        for target in targets:
            print(f"  - {target}")
        if only_print_targets:
            return

    if not yes:
        response = input("\nAre you sure you want to proceed? [y/N] ").strip().lower()
        if response not in ["y", "yes"]:
            print("Aborted.")
            return

    for target in targets:
        if os.path.exists(target):
            if not quiet:
                print(f"Removing {target}/ ...")
            shutil.rmtree(target, ignore_errors=True)
        else:
            if not quiet:
                print(f"{target}/ does not exist, skipping.")

    if not quiet:
        print("Clean-up completed.")

def main():
    parser = argparse.ArgumentParser(description="Clean up generated files in the project.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt (force clean).")
    parser.add_argument("-q", "--quiet", action='store_true', help="Be quiet.")
    parser.add_argument("-p", "--print_targets_only", action='store_true', help="Only print the list of targets for deletion without actually deleting them.")
    
    args = parser.parse_args()

    clean_project(
        yes=args.yes,
        quiet=args.quiet,
        only_print_targets=args.print_targets_only
    )

if __name__ == "__main__":
    main()
