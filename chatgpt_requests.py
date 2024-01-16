from openai import OpenAI
key = "" #put api key here


def request_to_chatgpt(content, model="gpt-3.5-turbo", n=3):
    client = OpenAI(api_key=key)

    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
      ], n=n
    )
    
    return [completion.choices[i].message.content for i in range(n)]
