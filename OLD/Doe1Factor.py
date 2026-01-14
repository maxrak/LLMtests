import statistics
from pprint import pprint


def get_app_ids(data):
    return set(item["apps"]["id"] for item in data if "apps" in item)

def count_versions_by_app(data, app_id, version):
    return sum(
        1
        for item in data
        if item.get("apps", {}).get("id") == app_id
        and item.get("version") == version
    )

def get_iterations_by_app_and_version(data, app_id, version):
    return [
        item.get("iterations")
        for item in data
        if item.get("apps", {}).get("id") == app_id
        and item.get("version") == version
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
    for version in ["v1","v2","v3"]:
        count = count_versions_by_app(data, app_id,version)
        iters = get_iterations_by_app_and_version(data,app_id,version)
        failed= iters.count(30)
        Okiters=remove_value(iters,30)
        #print(app_id+" "+version+ "\n\tcount:"+str(count)+ "\n\titers: "+ str(iters))
        if(len(Okiters)>1):
            avg,std=media_e_deviazione(Okiters)
        else:
            avg=Okiters[0]
            std=0
        #print(app_id+" "+version+ " count:"+str(count)+ " failed:"+str(failed)+" iters(avg): "+ f"{avg:.3f}"+ " iters(std): "+ f"{std:.3f}")
        result[version]=[count,failed,avg,std]
    return result

def printResults(appid, result):
    for version in result:
        data=result[version]
        print(appid+" "+version+ " count:"+str(data[0])+ " failed:"+str(data[1])+" iters(avg): "+ f"{data[2]:.3f}"+ " iters(std): "+ f"{data[3]:.3f}")

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

    
