import requests
import json
import statistics
from pprint import pprint


def sessions():
    url = "https://database.vseclab.lan/api/v2/tables/mmzjs8j9yb88tsf/records?offset=0&limit=25&where=&viewId=vwpkfrf4ivtd846e"

    headers = {"xc-token": "7x4DdSHBSIeVKVv_sRHW7Dz_VJ1bf8SkXzRJMLz0"}

    page=1
    rows=[]
    alldone=False

    while (not alldone):
        params = {
            "page": page,
            "pageSize": 25   # oppure 500, 1000 se consentito
        }
        response = requests.get(url, headers=headers, params=params, verify="public.crt")
        data=response.json()
        rows.extend(data["list"])
        #print(json.dumps(data, indent=2, ensure_ascii=False))
        page+=1
        if data["pageInfo"]["isLastPage"]:
            alldone=True

    return rows

def get_app_ids(data):
    return set(item["apps"]["id"] for item in data if "apps" in item)

def count_versions_by_app(data, app_id, version,summarized):
    return sum(
        1
        for item in data
        if item.get("apps", {}).get("id") == app_id 
        and item.get("apps", {}).get("summarized") == summarized
        and item.get("version") == version
    )

def get_iterations_by_app_and_version(data, app_id, version,summarized):
    return [
        item.get("iterations")
        for item in data
        if item.get("apps", {}).get("id") == app_id
        and item.get("version") == version
        and item.get("apps", {}).get("summarized") == summarized
        and "iterations" in item
    ]

def media_e_deviazione(valori):
    media = statistics.mean(valori)
    std = statistics.stdev(valori)      # deviazione standard campionaria
    return media, std

def remove_value(lista, valore):
    return [x for x in lista if x != valore]

def appResult(data,app_id):
    result={}
    print(data)
    for version in ["v1","v2","v3"]:
        for summarized in [0,1]:
            count = count_versions_by_app(data, app_id,version,summarized)
            iters = get_iterations_by_app_and_version(data,app_id,version,summarized)
            failed= iters.count(30)
            Okiters=remove_value(iters,30)
            print(app_id+" "+version+ "\n\tcount:"+str(count)+ "\n\titers: "+ str(iters))
            if(len(Okiters)>1):
                avg,std=media_e_deviazione(Okiters)
            elif(len(Okiters)==1):
                avg=Okiters[0]
                std=0
            else:
                avg=-1
                std=-1
            #print(app_id+" "+version+ " count:"+str(count)+ " failed:"+str(failed)+" iters(avg): "+ f"{avg:.3f}"+ " iters(std): "+ f"{std:.3f}")
            result[(version,summarized)]=[count,failed,avg,std]
    return result

def printResults(appid, result):
    for (version,summarized) in result:
        data=result[(version,summarized)]
        if summarized == 0: 
            sum="Not Summarized"
        else:
            sum="Summarized"
        print(appid+" "+version+" "+sum+ " count:"+str(data[0])+ " failed:"+str(data[1])+" iters(avg): "+ f"{data[2]:.3f}"+ " iters(std): "+ f"{data[3]:.3f}")

def Allresults(data):
    app_ids = get_app_ids(data)
    apps={}
    for appid in app_ids:
        res=appResult(data,appid)
        apps[appid]=res
    return apps

def printAll(apps):
    for app in apps:
        result=apps[app]
        printResults(app,result)

    
