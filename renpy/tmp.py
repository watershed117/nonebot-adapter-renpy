import openai

client = openai.OpenAI(api_key='937041ab8a8913329177a408cc96fd4b.lC8CldSAjskjGtgS',
                       base_url="https://open.bigmodel.cn/api/paas/v4")

response = client.chat.completions.create(messages=[{"role":"user","content":"你好"}], model="glm-4-flash-250414")
print(type(response.choices[0].message))
print(response.choices[0].message.content)