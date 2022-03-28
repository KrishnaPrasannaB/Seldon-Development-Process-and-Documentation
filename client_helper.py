import docker
from time import sleep
# Create container from image.
# Localhost port: 3000, tcp container port: 9000
def create_container():
    client = docker.from_env()
    container = client.containers.run("wine-classifier-seldon-image", detach=True,
                                      volumes=['/Users/p70002x/PycharmProjects/WineClassifierProject:/app'],
                                      ports={'9000/tcp': ('127.0.0.1', 3000)})

    sleep(2)
    print("Container ", container, " created successfully")
    return container

def stop_container(container):
    container.stop()

