Vagrant.configure(2) do |config|

  # The name of the Vagrant box to be used as the template for VMs
  config.vm.box = "ubuntu/jammy64"
  #config.vm.box_check_update = true

  config.vm.provider "virtualbox" do |vb|
    # Display the VirtualBox GUI when booting the machine
    vb.gui = true
    # Customize the amount of memory on the VM:
    vb.memory = "8192"
    vb.customize ["modifyvm", :id, "--usb", "on"]
    vb.customize ["modifyvm", :id, "--clipboard", "bidirectional"]

    # Without this, the serial console is slow.
    vb.customize ["modifyvm", :id, "--uart1", "0x3F8", "4"]
    vb.customize ["modifyvm", :id, "--uartmode1", "file", File.join(Dir.pwd, "serial-console.log")]

    if Vagrant.has_plugin?("vagrant-disksize")
      config.disksize.size = '20GB'
    end
  end

  # Port forwardings
  #config.vm.network "forwarded_port", guest: 1099, host: 1099

  # Create a private network, which allows host-only access to the machine
  # using a specific IP.
  #config.vm.network "private_network", ip: "192.168.187.188"

  # Share an additional folder to the guest VM. The first argument is
  # the path on the host to the actual folder. The second argument is
  # the path on the guest to mount the folder. And the optional third
  # argument is a set of non-required options.
  # config.vm.synced_folder "../data", "/vagrant_data"
  # DISABLED because it led to problems in some cases (possibly when the src directory contains the subdirectory with this Vagrantfile).
  #if defined?(HOST_SRC_DIR)
  #  config.vm.synced_folder HOST_SRC_DIR, "/src"
  #end

  # Install packages
  config.vm.provision "shell", inline: <<-SHELL
    # (From https://www.siegescape.com/blog/2019/4/14/speeding-up-vagrant-with-apt-mirror-uris)
    echo "Setting up apt to use fast mirrors and skip sources repos..."
    #sed -i 's|deb http://archive.ubuntu.com.ubuntu|deb mirror://mirrors.ubuntu.com/mirrors.txt|g' /etc/apt/sources.list
    sed -i 's|deb http://archive.ubuntu.com.ubuntu|deb https://mirrors.xtom.de/ubuntu|g' /etc/apt/sources.list
    sed -i '/deb-src/d' /etc/apt/sources.list

    wget -qO - https://packagecloud.io/AtomEditor/atom/gpgkey | apt-key add -
    echo "deb [arch=amd64] https://packagecloud.io/AtomEditor/atom/any/ any main" > /etc/apt/sources.list.d/atom.list

    apt-get update
    apt-get upgrade --yes --quiet

    PACKAGES=""
    # JDK is required to run the Gradle build system
    PACKAGES="${PACKAGES} openjdk-17-jdk"
    # Git is required to pull and push the source code
    PACKAGES="${PACKAGES} git"
    # tig helps with browsing the Git history in a terminal, kdiff3 with merging
    PACKAGES="${PACKAGES} tig kdiff3"
    # PlantUML is optional for rendering UML diagrams
    PACKAGES="${PACKAGES} plantuml"
    # A desktop environment and a terminal emulator
    PACKAGES="${PACKAGES} ubuntu-mate-desktop terminator"
    # Synaptic provides a GUI for package management
    PACKAGES="${PACKAGES} synaptic"

    # Some basic command line tools
    PACKAGES="${PACKAGES} bc bash patch gzip bzip2 tar cpio unzip rsync wget curl parted"
    # Some basic development tools
    PACKAGES="${PACKAGES} debianutils sed make binutils build-essential cmake"
    # Compilers and interpreters
    PACKAGES="${PACKAGES} gcc gcc-multilib g++ g++-multilib perl python2.7 python2 python-is-python3 ccache"
    # Development frameworks/libraries and associated tools
    PACKAGES="${PACKAGES} libncurses5-dev libboost-all-dev libfmt-dev libspdlog-dev libsqlite3-dev"
    # Qt 5 and tools, including the designer
    PACKAGES="${PACKAGES} qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools qttools5-dev-tools"
    # The capabilities library
    PACKAGES="${PACKAGES} libcap-dev"
    # Code analysis and code quality tools and their dependencies
    PACKAGES="${PACKAGES} cppcheck cppcheck-gui valgrind astyle doxygen-gui graphviz iwyu"
    # An MQTT broker for testing
    PACKAGES="${PACKAGES} mosquitto mosquitto-clients"
    # QEMU for testing images, including UEFI support
    PACKAGES="${PACKAGES} qemu-system qemu-system-gui ovmf"
    # Fltk for some older UI classes
    PACKAGES="${PACKAGES} libfltk1.3-dev"
    # matplotlib and numpy are used for Buildroot's graph generation feature
    PACKAGES="${PACKAGES} python3-matplotlib python3-numpy"
    # libssl-dev is required for the multiwall-backend build
    PACKAGES="${PACKAGES} libssl-dev"
    # is required for pointCloudLib
    PACKAGES="${PACKAGES} libxext-dev libx11-dev x11proto-gl-dev libglvnd-dev liblz4-dev"

    # Text editor
    PACKAGES="${PACKAGES} atom"
    # C++ IDE
    PACKAGES="${PACKAGES} qtcreator"

    # OpenCV for detecting a vehicle by its characteristic colors.
    PACKAGES="${PACKAGES} libopencv-dev"

    # Select exactly one display manager.
    echo gdm shared/default-x-display-manager select lightdm | debconf-set-selections
    echo lightdm shared/default-x-display-manager select lightdm | debconf-set-selections

    DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet ${PACKAGES}

    # Ensure Clang 8 is used to avoid errors in Qt Creator.
    #update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-8 100
    #update-alternatives --install /usr/bin/clang clang /usr/bin/clang-8 100
  SHELL

  config.vm.provision "shell", privileged: false, inline: <<-SHELL
    echo "Disabling screensaver and screen lock..."
    gsettings set org.mate.screensaver idle-activation-enabled false
    gsettings set org.mate.screensaver lock-enabled false
  SHELL

end
