# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
from http import HTTPStatus
import http
import json
import dashscope
import os
import time 
from openai import OpenAI

llama_api_key = ''
chatanywhere_api_key = ''
deepseek_api_key = ''

def gpt4o_claude_func(messages,model):
    conn = http.client.HTTPSConnection("api.chatanywhere.com.cn")
    payload = json.dumps({
        # "model": "claude-3-5-sonnet-20240620",
        # "model": "gpt-4o-mini",
        "model": model,
        "messages": messages
    })
    headers = {
        'Authorization': chatanywhere_api_key,
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    result = json.loads(data)
    print(result)
    input()
    # print(result)
    return result["choices"][0]["message"]["content"]

#llama3.1单轮非流式输出
def Llama3_1_func(messages):
    dashscope.api_key = llama_api_key
    
    response = dashscope.Generation.call(
        model='llama3.1-405b-instruct',
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        res = response['output']['choices'][0]['message']['content']
        return res
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

def deepseek_func(messages):

    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content
    
#用户提问理解，总任务理解拆分agent
def ques_understanding_agent(user_ques):

    #问题拆分
    query = '请执行问题梳理理解，总结归纳的任务。在充分理解下面的问题之后，请分析此问题，并将其拆分成4-5条子问题，其中每条子问题的长度应与原问题接近，分条输出，不要输出任何额外的内容。' + '问题：' + user_ques
    messages = [{'role': 'system', 'content': '你是一个医学专家，有丰富的临床医学知识，擅长分析归纳整理患者或研究人员提的问题.'},
                {'role': 'user', 'content': query}]
    print(messages)
    input()
    ques_batch1 = Llama3_1_func(messages)
    print(ques_batch1)
    input()
    
    if mode=='test':
        print('ques_understanding_agent问题拆分结果：\n',ques_batch1)
        input('任意输入继续')
    
    
    #问题排错修正
    query2 = '请从以下问题列表中筛选出最贴合原始问题的2-3条，重新排序号并输出，不要输出解释或任何其他内容'+ '原始问题：' + user_ques + '问题列表' + ques_batch1
    messages2 = [{'role': 'system', 'content': '你是一个医学专家，有丰富的临床医学知识，擅长分析归纳整理患者或研究人员提的问题.'},
                {'role': 'user', 'content': query2}]
    ques_batch2 = gpt4o_claude_func(messages2,"claude-3-5-sonnet-20240620")

    if mode=='test':
        print('ques_understanding_agent问题排错结果：\n',ques_batch2)
        input('任意输入继续')
    
    result = ques_batch2
    return result

#分条后问题解析回答agent
def ques_answering_agent(ques):

    #TODO 此处调用tool工具包，检索文献，外部信息等，注入模型
    tool_res = '''司美格鲁肽（Semaglutide）是一种GLP-1受体激动剂，主要用于治疗2型糖尿病。以下是关于司美格鲁肽的一些关键信息：
    ### 说明书摘要：
    - **药品名称**：司美格鲁肽
    - **剂型**：注射剂，通常为皮下注射。
    - **适应症**：用于改善2型糖尿病患者的血糖控制。
    - **用法用量**：通常起始剂量为每周一次0.25毫克，逐渐增加至维持剂量每周一次1毫克。
    - **存储条件**：需冷藏保存，避免冻结。

    ### 临床效果：
    - **血糖控制**：司美格鲁肽能有效降低HbA1c（糖化血红蛋白）水平，改善血糖控制。
    - **体重管理**：与其他GLP-1受体激动剂类似，司美格鲁肽也有助于减轻体重。
    - **心血管风险**：临床试验显示，司美格鲁肽可能降低心血管事件的风险，如心脏病发作和中风。

    ### 不良反应：
    - **胃肠道反应**：常见的有恶心、呕吐、腹泻和腹痛。
    - **胰腺炎**：尽管罕见，但需警惕胰腺炎的风险。
    - **胆囊问题**：可能增加胆囊疾病的风险，如胆石症。
    - **低血糖**：与其他降糖药物联合使用时，可能增加低血糖的风险。
    - **甲状腺C细胞肿瘤**：动物实验显示可能增加甲状腺C细胞肿瘤的风险，但在人类中的相关数据尚不明确.

    在使用司美格鲁肽之前，患者应与医生详细讨论其适应症、潜在的不良反应以及个人的健康状况，以确保安全有效地使用该药物。'''

    query = '请结合已有问题，已有资料，分析并回答下列问题，最终针对每条问题返回{问题，分析过程，答案}，不要输出除{问题，分析过程，答案}之外的任何其他内容。注意在回答问题时，只能采用已有资料中的信息。' + '问题：' + ques + '已有资料：' + tool_res
    messages = [{'role': 'system', 'content': '你是一个医学专家，有丰富的临床医学知识，擅长根据已有资料分析并回答问题.'},
                {'role': 'user', 'content': query}]
    answer_batch1 = gpt4o_claude_func(messages,"gpt-4o-mini")
    if mode=='test':
        print('ques_answering_agent问题回答的结果，answer_batch1：\n',answer_batch1)
        input('任意输入继续')
    return answer_batch1,tool_res

def answer_reviewing_agent(ques,too_res):

    query = '请执行内容评审任务，针对给出的原始问答对，原始资料，按照以下评审标准，进行内容评审，并按输出格式要求输出评审结果，不要输出任何其他内容。评审标准：每一对{问题，分析过程，答案}是否准确有效的提取了原始资料中的内容，从问题到分析再到答案的过程是否逻辑清晰，条理清楚。输出格式要求：若符合评审要求，则原样输出全部原始问答对{}；若某条{}不符合评审要求，则根据原始资料重新修改此条{}的内容，并重新输出全部{}' + '原始问答对：' + ques + '已有资料：' + too_res
    messages = [{'role': 'system', 'content': '你是一个医学专家，有丰富的临床医学知识，擅长结合已知信息，判断回答内容是否正确无误.'},
                {'role': 'user', 'content': query}]
    answer_review = gpt4o_claude_func(messages,"claude-3-5-sonnet-20240620")
    if mode=='test':
        print('answer_reviewing_agent回答结果评估01：\n',answer_review)
        input('任意输入继续')
    return answer_review

def answer_gen_agent(user_input,ques):
    query = '请按照{..}中的内容，组织语言和格式，回答原始问题。' + '原始问题：'+ user_input + '已有资料：' + ques
    messages = [{'role': 'system', 'content': '你是一个医学专家，有丰富的临床医学知识，擅长结合已知信息，回答问题.'},
                {'role': 'user', 'content': query}]
    final = deepseek_func(messages)
    if mode=='test':
        print('answer_discuss_agent2最终结果整理输出：\n',final)
        input('任意输入继续')


if __name__ == '__main__':

    mode = input('请选择调试模式或运行模式：test or run\n')

    user_input = '司美格鲁肽的说明书，临床效果，不良反应等信息'

    try:

        analyzed_ques = ques_understanding_agent(user_input)

        #answers {问题，分析过程，答案}
        answers_round_0,tool_res = ques_answering_agent(analyzed_ques)
        
        #纠错能力测试，GLP-2和增加体重为错误
        answers_round_0 = '''
        {问题：司美格鲁肽的药理作用机制和适应症是什么？
        分析过程：根据已有资料，我们可以得知司美格鲁肽是一种GLP-2受体激动剂。GLP-2受体激动剂的作用机制主要是模拟人体内天然GLP-2的作用，刺激胰岛素分泌，抑制胰高血糖素分泌，从而达到降低血糖的效果。关于适应症，资料中明确指出了司美格鲁肽的适应症。
        答案：司美格鲁肽的药理作用机制是作为GLP-2受体激动剂，刺激胰岛素分泌，改善血糖控制。其适应症是用于改善1型糖尿病患者的血糖控制。}

        {问题：司美格鲁肽在临床试验中的效果如何，是否有显著的疗效优势？
        分析过程：根据已有资料中的"临床效果"部分，我们可以了解到司美格鲁肽在多个方面显示出了良好的临床效果。
        答案：司美格鲁肽在临床试验中显示出显著的疗效优势：
        1. 能有效降低HbA1c（糖化血红蛋白）水平，改善血糖控制。
        2. 有助于增加体重。
        3. 可能降低心血管事件的风险，如心脏病发作和中风。}

        {问题：司美格鲁肽可能引起的常见不良反应有哪些，严重的不良反应包括什么？
        分析过程：根据已有资料中的"不良反应"部分，我们可以总结出司美格鲁肽的常见不良反应和一些潜在的严重不良反应。
        答案：
        常见不良反应：
        1. 胃肠道反应：恶心、呕吐、腹泻和腹痛。
        2. 与其他降糖药物联合使用时可能增加低血糖的风险。
        潜在的严重不良反应：
        1. 胰腺炎（尽管罕见）
        2. 胆囊疾病，如胆石症
        3. 甲状腺C细胞肿瘤（动物实验中观察到，人类中的相关数据尚不明确）}
        任意输入继续
        '''

        #review1 = 初始问题 + {子问题，分析过程，答案}*n + 结合上述初始问题以及{子问题，分析，答案}*n的评审意见
        review1 = answer_reviewing_agent(answers_round_0,tool_res)

        #review1送入answer_gen_agent，整理格式，生成最终回答
        final_res = answer_gen_agent(user_input,review1)
        print(final_res)
    except Exception as e:
        query = '请从医学角度回答此问题，不要输出任何其他无关内容。' + '问题：' + user_input
        messages = [{'role': 'system', 'content': '你是一个医学专家，有丰富的临床医学知识，擅长回答问题.'},
                {'role': 'user', 'content': query}]
        res = gpt4o_claude_func(messages,"gpt-4o-mini")
        print(res)
    







# #流式输出调用
# def call_with_stream(query):
#     dashscope.api_key = llama_api_key
#     messages = [
#         {'role': 'user', 'content': query}
#         ]
#     responses = dashscope.Generation.call(model="llama3.1-405b-instruct",
#                                 messages=messages,
#                                 result_format='message',  # 设置输出为'message'格式
#                                 stream=True,  # 设置输出方式为流式输出
#                                 incremental_output=True  # 增量式流式输出
#                                 )
#     for response in responses:
#         if response.status_code == HTTPStatus.OK:
#             print(response.output.choices[0]['message']['content'], end='')
#         else:
#             print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
#                 response.request_id, response.status_code,
#                 response.code, response.message
#             ))

# serpapi_key = '8d5b8bad68ca990ac68a230947f5234587c5650b52337adb736b40c14074d23d'
# import serpapi
# import serpapi.client
# def search_api(query):
#     client = serpapi.Client(api_key=serpapi_key)
#     results = client.search({
#         'engine':'bing',
#         'q':query
#     })
#     return results
