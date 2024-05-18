import os
import json
from dotenv import load_dotenv
import jwt    # PyJWT
import uuid
import hashlib
import requests
from urllib.parse import urlencode

load_dotenv()

access_key = os.getenv('ACCESS_KEY')
secret_key = os.getenv('SECRET_KEY')
base_url = os.getenv('BASE_URL')


def upbit_pre(query=None):
    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
    }

    if query is not None:
        m = hashlib.sha512()
        m.update(urlencode(query).encode())
        query_hash = m.hexdigest()

        payload['query_hash'] = query_hash
        payload['query_hash_alg'] = 'SHA512'

    jwt_token = jwt.encode(payload, secret_key)
    authorization_token = 'Bearer {}'.format(jwt_token)
    return authorization_token


def upbit_get(postfix, query=None):
    token = upbit_pre(query)
    return requests.get(base_url + postfix, headers={'Authorization': token}, params=query)


def upbit_post(postfix, query=None):
    token = upbit_pre(query)
    return requests.post(base_url + postfix, headers={'Authorization': token}, json=query)


def pjson(data):
    jstr = json.dumps(data, indent=4)
    print(jstr)