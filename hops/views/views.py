from django.template import loader
from django.http import HttpResponse

def index(request):
    # latest_question_list = Question.objects.order_by('-pub_date')[:5]
    template = loader.get_template('hops/hops.html')
    context = {}
    return HttpResponse(template.render(context, request))