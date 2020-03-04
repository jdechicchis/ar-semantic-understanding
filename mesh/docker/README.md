# Usage Instructions

Run `docker build -t python-pcl:latest . --network host` to setup the docker container. Use `docker run -t -i -v /path/from/host:/mnt python-pcl` to run the container and mount `/path/from/host` to `/mnt` on the container.

## GUI Support

For Mac see: https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc

xhost +

Test: 
sudo docker run -t -i -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/Downloads:/mnt gns3/xeyes

docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v `pwd`:/mnt pymesh bash
