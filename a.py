for package in $(cat packages.txt); do pip uninstall -y $package; done

