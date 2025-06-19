    #
    #
    # ############################################## #
    # #  splatops/cntn-signalbackup-tools/README.md #
    # ############################################ #
    #
    # Brought to you by...
    # 
    # ::::::::::::'#######::'########:::'######::
    # :'##::'##::'##.... ##: ##.... ##:'##... ##:
    # :. ##'##::: ##:::: ##: ##:::: ##: ##:::..::
    # '#########: ##:::: ##: ########::. ######::
    # .. ## ##.:: ##:::: ##: ##.....::::..... ##:
    # : ##:. ##:: ##:::: ##: ##::::::::'##::: ##:
    # :..:::..:::. #######:: ##::::::::. ######::
    # ::::::::::::.......:::..::::::::::......:::
    #
    ##################################################
   

# Containerizing signalbackup-tools
SplatOps presents a contribution to the signalbackup-tools project for building and running in a container.  All examples use and assume the end user has installed Docker and supporting tools.  The easiest way to install Docker outside of native OS packages can be found here: [Get Docker](https://get.docker.com/).

Link to the original project: [signalbackup-tools on Github](https://github.com/bepaald/signalbackup-tools)

## Getting Started
These instructions will walk you through:

* Building the signalbackup-tools Docker image
* Running the signalbackup-tools Docker image

## Why
Most people have a basic understanding of what containers are.  The advantage of packaging signalbackup-tools into a container include:
* Being able to run the tool natively on a Debian based system without having to munge with a difference in out of the box library support.  For example, at the time this was written, the `crypto++` library requires version 8.2.0, but all Debian/Ubuntu systems only have version 5.6.4 available.
* Being able to quickly build the container, run the tool, and dispose of everything you've just done without tainting any portion of your system.
* Spend your time debugging a failed Signal backup vs building a tool.

### Building the signalbackup-tools Docker Image Locally
Clone this repository. Then...
* ```cd cntn-signalbackup-tools```
* ```docker build -t signalbackuptools:latest .```
  
To validate signalbackup-tools is in your local container images...
* ```docker image ls | grep signalbackuptools```

### Running the signalbackup-tools Docker Image
The signalbackup-tools binary is run as a standalone tool against local files which generally include a Signal database backup.  With this in mind we  want the container to work against a local copy of the file without having to do any file copying to the running container or deal with creating explicit mount paths.  To solve for this we can leverage `$PWD` in a pattern wherein the container will work on the path the container starts in:
* First, switch to the directory on your machine (this is assuming a Linux/UNIX OS) that contains your Signal backup file:
  * ```cd /home/user/mysignalbackup```
* Finally run the Docker container with the appropriate flags for signalbackup-tools.  This example uses the example from the signalbackup-tools README:
  * ```docker run -it -v "$PWD:$PWD" -w "$PWD" signalbackuptools:latest signal-2020-01-01-01-01-01.backup 000000000000000000000000000000 --output signal-2020-01-01-01-01-01-fixed.backup --opassword 000000000000000000000000000000```

#### FAQ

* So I don't need to tell the container to run the `signalbackup-tools` binary?
  * No. If you look at the Dockerfile the ENTRYPOINT is the `signalbackup-tools` command.  That means everything passed to the container as arguments will be automagically passed to the binary when the container starts.
* How does the container access my local files?
  * We use a bit of a container runtime hack where the current directory is passed to the container and the container mimics that in the container, this happens with the `-v` option flag (bind mount flag).  Then we set the working directory in the container to our working directory in the same manner, this happens with the `-w` flag (working directory flag).
* Where does my new output file get written?
  * The local directory you're in when you start the container. For simplicity sake you'll want to keep the source and destination in the same directory, as directory structure outside of the parent path of your working directory isn't known to the running container.
* I'm trying to build a new version of the container, but I keep getting an old version and the build finishes almost immediately.  What gives?
  * Docker doesn't know when a Git repository changes, so you need to tell it to ignore the build cache and force the build to clone the repo again.  You can do this a variety of ways within the Dockerfile, however the easiest way to work around this is to tell Docker to ignore the build cache during the build command.  Like this:
    * ```docker build --no-cache -t signalbackuptools:latest .```
* I want to rebuild the container image because I know the GitHub repo has been updated, but I don't want to pull all of the OS updates down again.  Help me?
  * Sure!  The DOCKERFILE is structured in a way that there are two RUN lines.  This results in two unique container layers.  The first layer does all of the OS packaging downloading and installation and the second does source code download and compilation.  All you need to do is change the first `echo` line in the `CODE BUILD` section by incrementing the default of `000`.  In fact, you can change anything in that line and it will force that layer to rebuild.  But make sure if you only want this one layer to be rebuilt you don't use the `--no-cache` option!  This will force all layers to be rebuilt.
* What if I want to do this from inside the container shell but still have access to my local directory I'm currently in on my local machine?
  * This is very similar, but we just need to tell Docker to ignore our ENTRYPOINT from the DOCKERFILE and instead run a shell.  You can do this with the following command which will drop you into a bash shell in the container and with the PATH set correctly for the `signalbackup-tools` binary and you'll have full R/W access to the local machine files you started the container from:
    * ```docker run -it -v "$PWD:$PWD" -w "$PWD" --entrypoint /bin/bash signalbackuptools:latest```