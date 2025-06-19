#
#
# ############################################### #
# #  splatops/cntn-signalbackup-tools/Dockerfile #
# ############################################# #
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
# CONTAINER BASE
FROM fedora:41

# VARIABLES
ENV \
    # SIGNALBACKUP-TOOLS VARIABLES
    S_PATH=/opt/signalbackup-tools \
    S_GHUB=https://github.com/bepaald/signalbackup-tools.git

# SET PATH AFTER S_PATH
ENV PATH=${PATH}:${S_PATH}

# ENVIRONMENT BUILD
RUN \
echo "### Install dependencies ###" && \
dnf -y install \
    gcc-g++ \
    openssl-devel \
    sqlite-devel \
    dbus-devel \
    which \
    git && \
echo "### Setup build structure ###" && \
mkdir -p ${S_PATH}

#CODE BUILD
RUN \
echo "### Increment me to force this container layer to be rebuilt: 000" && \
echo "### Retrieve signalbackup-tools source ###" && \
git clone ${S_GHUB} ${S_PATH} && \
echo "### Make signalbackup-tools ###" && \
cd ${S_PATH} && \
bash BUILDSCRIPT.bash

# CONTAINER STAGING
WORKDIR ${S_PATH}
ENTRYPOINT [ "signalbackup-tools" ]