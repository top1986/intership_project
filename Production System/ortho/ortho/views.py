from django.shortcuts import render
from .forms import PredictForm
#import json
# import yaml

def home(request):
    # with open('spec_files/attributes.yaml', 'r') as attrs:
    #     attributes = yaml.load(attrs)

    if request.method == 'POST':
    	form = PredictForm(request.POST)



    	if form.is_valid():
    		#json_data = json.loads(str(request.body.decode(encoding='UTF-8')))
    		return render(request, 'pages/success.html', {'resp':request.POST.dict()})
    else:
    	form = PredictForm() 
        
    return render(request, 'home.html', {'form': form})

def evaluation(request):
    return render(request, 'pages/evaluation.html', None)

def monitoring(request):
    return render(request, 'pages/monitoring.html', None)


def management(request):
    return render(request, 'pages/management.html', None)

def success(request):
    return render(request, 'pages/success.html', {'decision':True})
