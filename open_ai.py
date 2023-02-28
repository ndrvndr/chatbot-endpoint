import openai
openai.api_key = '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'

while True:
    prompt = input('Enter new prompt: ')
    if 'exit' in prompt or 'quit' in prompt:
        break
    
    completion = openai.Completion.create(
        model = "text-davinci-003",
        prompt = prompt,
        max_tokens = 500,
        temperature = 0,
    )
    
    response = completion.choices[0].text
    print(response)