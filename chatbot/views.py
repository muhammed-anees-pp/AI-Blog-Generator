from django.shortcuts import render
from django.http import JsonResponse
from huggingface_hub import InferenceClient
from django.conf import settings

HT_API_KEY = settings.HT_API_KEY
client = InferenceClient(api_key=HT_API_KEY)

def query_hf_model(user_input):
    try:
        prompt = f"You are a helpful AI assistant. Respond to: {user_input}"
        response = client.text_generation(
            prompt,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            max_new_tokens=500,
            temperature=0.7,
        )
        return response.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        response = query_hf_model(message)
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html')