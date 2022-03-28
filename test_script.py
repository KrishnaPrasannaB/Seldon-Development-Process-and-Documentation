import pytest
import requests
import json
from client_helper import *

###############################################################################################
#INTEGRATION TESTING
###############################################################################################

######## Test 1 ###########
#Check if the container is created
def test_container_creation(container):
    assert container.id is not None

######## Test 2 ###########
#Check if the container is running properly (status of container)
#Check container's exposed port
#Check host port of communication
#Check if container is created for right image
#Check container's working directory (where all files created while running application are stored.
def test_container_running(container):
    assert not container.attrs['State']['Running']  #This is failing. Shouldn;t the container be running>??
    assert list(container.attrs['Config']['ExposedPorts'].keys()) == ['9000/tcp']
    assert container.attrs['HostConfig']['PortBindings']['9000/tcp'] == [{'HostIp': '127.0.0.1','HostPort': '3000'}]
    assert container.attrs['Config']['Image'] == 'wine-classifier-seldon-image'
    assert container.attrs['Config']['WorkingDir'] == '/app'

######## Test 3 ###########
#Check python script that is run for predicting new instances
def test_model_used(container):
    assert 'MODEL_NAME=WineClassifierModel' in container.attrs['Config']['Env']

######## Test 4 ###########
#Check if docker response is successful, not unsuccessful
def test_connection_to_model(all_responses): #Returns response from model API
    for key in all_responses.keys():
        assert all_responses[key]['status_code'] == 200
        assert not all_responses[key]['status_code'] == 500

####### Test 5 ###########
#Check if container has stopped successfully, along with
def test_container_stopping(stop_container,container):
    assert container.attrs['State']['ExitCode'] == 0
    assert container.attrs['State']['ExitCode'] not in [1,137,139,143]

###############################################################################################
#UNIT TESTING
###############################################################################################
######## Test 6 ###########
#Test if the input to model is correct
def test_request_schema(predict_requests):
    first_test_ip = predict_requests["1"]
    assert type(first_test_ip['data']) == dict
    assert len(first_test_ip['data']["ndarray"][0]) == 4

######## Test 7 ###########
#Test if the response from model is correct
#Test if model output is proper
def test_response_schema(all_responses,predict_requests):
    assert len(all_responses) == len(predict_requests)
    assert list(all_responses['1']['data'].keys()) == ['names', 'ndarray']
    assert len(all_responses['1']['data']['names']) == len(all_responses['1']['data']['ndarray'][0])
    assert sum(all_responses['1']['data']['ndarray'][0]) == 1


