import requests
import json
from multiprocessing import Pool

def get_protocol_data(p_id):
    try:
        protocol_obj = \
        requests.get('https://www.protocols.io/api/v3/protocols/%s'%p_id, 
                         headers = {'Authorization': 'Bearer d60fc3b2cd61bcef7782d8425d25d68b4d6193c5a5cf3560ea10aae2cc0ca64d'}).json()['protocol']
        with open('data/%s.json'%p_id, 'w') as f:
            json.dump(protocol_obj, f, indent = 4)
    except Exception as e:
        print(e)
        pass



pl = Pool(8)
page = 1
page_size = 100
while(True):
    try:
        data = \
            requests.get('https://www.protocols.io/api/v2/protocols?filter="public"',
                     headers = {'Authorization': 'Bearer d60fc3b2cd61bcef7782d8425d25d68b4d6193c5a5cf3560ea10aae2cc0ca64d'},
                    params={'page_size' : page_size, 'page_id' : page}).json()
        if len(data['protocols']) == 0:
            break
        page +=1
        protocol_ids = [protocol['protocol_id'] for protocol in data['protocols']]
        pl.map(get_protocol_data, protocol_ids)
        print('crawled page %d'%page)
    except Exception as e:
        print(e)
        break



