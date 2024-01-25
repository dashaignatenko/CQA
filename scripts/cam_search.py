import requests
import json

def collect_arguments_for_cam(o1, o2, asp="", topn=10, weight=3):
    lt = "ltdemos.informatik.uni-hamburg.de"
    address = f"http://{lt}/cam-api/cam?model=default&fs=false&objectA={o1}&objectB={o2}"
    if asp:
        address+= f"&aspect1={asp}&weight1={weight}"
    x = requests.get(address)
    result = x.json()
    topn_1 = [i['text'] for i in result['object1']['sentences'][:topn]]
    topn_2 = [i['text'] for i in result['object2']['sentences'][:topn]]
    return topn_1, topn_2