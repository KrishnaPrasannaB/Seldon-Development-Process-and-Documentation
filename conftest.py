import pytest
import requests
import json
from client_helper import *


#Setting up fixtures for testing.
@pytest.fixture(scope="module")
def url():
    return 'http://127.0.0.1:3000/predict'


@pytest.fixture(scope="module")
def headers():
    return {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

@pytest.fixture(scope="module")
def predict_requests():
    f = open('predict_requests.json')
    return json.load(f)

@pytest.fixture(scope="module")
def container():
    container = create_container()
    return container

@pytest.fixture(scope="module")
def all_responses(predict_requests, url, headers): #Returns response from model API
    all_responses = {}
    key = 1
    while key <= len(predict_requests.keys()):
        payload = json.dumps(predict_requests[str(key)])
        api_response = requests.post(url, data=payload, headers=headers)
        all_responses[str(key)] = api_response.json()
        all_responses[str(key)]['status_code'] = api_response.status_code
        all_responses[str(key)]['url'] = api_response.url
        key += 1

    with open('all_responses', 'w') as datafile:
        json.dump(all_responses, datafile)
    #pprint(all_responses)
    return all_responses

@pytest.fixture(scope="module")
def stop_container(container):
    container.stop()
    print("Container stopped successfully")
    print(container.attrs['State'])

#To read the all_responses file
# f = open('all_responses')
# all_responses_file = json.load(f)
# pprint(all_responses_file)