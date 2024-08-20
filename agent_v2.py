import http
import json
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import random
import hashlib
import urllib
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

chatanywhere_api_key = ''


def baidu_fanyi(q):
    appid = '20230504001665427'
    secretKey = ''
    myurl = '/api/trans/vip/fieldtranslate'
    fromLang = 'auto'
    toLang = 'en'
    salt = random.randint(32768, 65536)
    domain = 'medicine'
    sign = appid + q + str(salt) + domain + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&domain=' + domain + '&sign=' + sign

    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
    httpClient.request('GET', myurl)

    response = httpClient.getresponse()
    result_all = response.read().decode("utf-8")
    try:
        result = json.loads(result_all)
    except:
        time.sleep(0.1)
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

    return result["trans_result"][0]["dst"]

# 模型翻译
def main_tmp_fanyi(query):
    conn = http.client.HTTPSConnection("api.chatanywhere.com.cn")
    payload = json.dumps({
        # "model": "claude-3-5-sonnet-20240620",
        # "model": "gpt-4o-mini",
        "model": "gpt-4o-mini",
        "messages": query
    })
    headers = {
        'Authorization': '',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    result = json.loads(data)
    return result["choices"][0]["message"]["content"]

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
    return result["choices"][0]["message"]["content"]

def gpt4o_claude_stream_func(messages,model):
    conn = http.client.HTTPSConnection("api.chatanywhere.com.cn")
    payload = json.dumps({
        # "model": "claude-3-5-sonnet-20240620",
        # "model": "gpt-4o-mini",
        "model": model,
        "messages": messages,
        "stream":"true"
    })
    headers = {
        'Authorization': chatanywhere_api_key,
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()

    # response_data = res.read().decode('utf-8')
    # print(response_data)
    # input()
    
    full_response = ""
    buffer = "" 
    # 读取流式响应
    while True:
        chunk = res.read(4096)
        if not chunk or chunk.endswith(b'data: [DONE]\n'):
            break  # 如果没有更多的数据或者接收到 [DONE]，则退出循环

        try:
            chunk_str = chunk.decode('utf-8')
        except UnicodeDecodeError as e:
            print(f"Unicode Decode Error: {e}")
            # 如果解码失败，尝试其他编码方式或处理不完整数据
            continue

        # 分割数据块为行
        lines = (buffer + chunk_str).split('\n')
        buffer = lines.pop()  # 最后一行可能不完整，保留下来

        # 处理每一行数据
        for line in lines:
            if line.startswith('data:'):
                data_str = line[6:]

                try:
                    data = json.loads(data_str)
                    delta_content = data['choices'][0]['delta'].get('content', '')
                    full_response += delta_content
                    print(delta_content, end='', flush=True)
                except json.JSONDecodeError as e:
                    # print(f"JSON Decode Error: {e}")
                    # 如果出现解码错误，将剩余的数据添加到下一个数据块
                    buffer += data_str
                    continue

    # 如果缓冲区中有剩余数据，尝试再次解析
    if buffer.startswith('data:'):
        data_str = buffer[6:]
        try:
            data = json.loads(data_str)
            delta_content = data['choices'][0]['delta'].get('content', '')
            full_response += delta_content
            print(delta_content, end='', flush=True)
        except json.JSONDecodeError as e:
            print(f"Final JSON Decode Error: {e}")
    
    # print("\nFull Response:", full_response)
    return full_response


#用户输入二分类，医学或药学
def ques_understanding_agent(user_ques):

    query = '请执行问题分类的任务，给你一个原始问题，不要回答此原始问题，而是对此问题做类别选择。判断原始问题是医学问题还是药学问题，并输出分类结果。输出格式为：医学问题，药学问题。例如判断为医学问题，则输出：医学问题。不要输出任何其他内容，不要任何解释或无关内容。原始问题：' + user_ques
    messages = [{'role': 'system', 'content': '你是一个医学专家，有丰富的临床医学知识，擅长分析患者或研究人员提的问题.'},
                {'role': 'user', 'content': query}]
    if mode =='test':
        print('\n问题二分类中。。。\n')
    ques_type_result = gpt4o_claude_func(messages,"claude-3-5-sonnet-20240620")
    if mode=='test':
        print('ques_understanding_agent问题分类结果：\n',ques_type_result)
        # input('\n任意输入继续\n')
        
    return ques_type_result

#按类别回答问题agent
def ques_answering_agent(user_ques,ques_type):

    #TODO 此处调用tool工具包，检索文献，外部信息等，注入模型
    query_start_time = time.time()
    tool_res,ids = retrieval(user_ques)
    tool_res = str(tool_res)
    query_end_time = time.time()
    r_time = query_end_time - query_start_time
    
    if mode=='test':
        print(tool_res)
    print('retrieval time:',r_time)

    if '药学问题' in ques_type:
        #药学处理流程
        query = '请参考指定的回答框架，结合外部资料与自身知识，回答用户问题，然后排版组织好格式后输出。\n回答框架：\n1.药品总述\n2.药品基本信息（化学式，CAS登录号，水溶性等等）\n3.适应症\n4.药理作用\n5.用法用量\n6.注意事项（慎用禁用特殊人群等）\n7.不良反应\n8.药物相互作用\n9.总结回答' +'\n外部资料：' + tool_res + '\n用户问题：' + user_ques +'\n输出要求：要求可以完全按照框架回答，也可以适当调整框架。要求在回答的对应位置标清楚所使用到的参考资料的序号，引用标号可以在任何位置，例如：司美格鲁肽是一种胰高血糖素样肽-1（GLP-1）受体激动剂[3]，主要用于2型糖尿病。'
        messages = [{'role': 'system', 'content': '你是一个医学专家，有丰富的临床药学知识，擅长根据自身知识和已有资料解答问题。'},
                    {'role': 'user', 'content': query}]
        if mode == 'test':
            print('\n药学问题回答生成中。。。。\n')
        answer = gpt4o_claude_func(messages,"gpt-4o-mini")
        # answer = answer + '\n以上信息仅供参考，如果有更多问题或需要进一步的医疗建议，请咨询专业医生。'
        if mode=='test':
            print('药学问题回答结果：\n',answer)
            # input('\n任意输入继续\n')
        return '药学问题',answer
    else:
        #医学类问题解答流程
        query = '请先思考回答原始问题最适合的组织结构，然后按照思考结果组织你的回答结构，按照参考资料与自身知识生成回答内容。使回答具有条理逻辑。只输出回答结果，不要输出任何分析和思考的过程。\n参考资料：' + tool_res + '\n输出要求：要求在回答的对应位置标清楚所使用到的参考资料的序号，引用标号可以在任何位置，例如：司美格鲁肽是一种胰高血糖素样肽-1（GLP-1）受体激动剂[3]，主要用于2型糖尿病。'
        messages = [{'role': 'system', 'content': '你是一个医学专家，有丰富的临床医学知识，擅长根据自身知识和已有资料解答问题。'},
                    {'role': 'user', 'content': query}]
        if mode == 'test':
            print('\n医学问题回答生成中。。。。\n')
        answer = gpt4o_claude_func(messages,"gpt-4o-mini")
        # answer = answer + '\n以上信息仅供参考，如果有更多问题或需要进一步的医疗建议，请咨询专业医生。'
        if mode=='test':
            print('ques_answering_agent问题回答的结果，answer_batch1：\n',answer)
            # input('\n任意输入继续\n')
        return '医学问题',answer

    return None

def answer_reviewing_agent(user_input,answer):

    query = '请对已有回答进行内容扩充和展开描述。要求：在已有回答的基础上做扩充，只输出扩充后的最终结果，不要输出任何其他内容。\n原始问题:' + user_input + '\n已有回答：' + answer
    messages = [{'role': 'system', 'content': '你是一个医学专家，有丰富的临床医学知识，擅长结合自身信息，进行修改扩充订正等.'},
                {'role': 'user', 'content': query}]
    if mode == 'test':
        print('\n回答评审修正中。。。。\n')
    answer_review = gpt4o_claude_func(messages,"claude-3-5-sonnet-20240620")
    answer_review = answer_review + '以上信息仅供参考，如果有更多问题或需要进一步的医疗建议，请咨询专业医生。'
    
    if mode=='test':
        print('answer_reviewing_agent回答结果评估：\n',answer_review)
        # input('\n任意输入继续\n')
    return answer_review

def retrieval(content):
    # 向量库连接
    ziiiiz_client = MilvusClient(
        uri="https://in01-e7e7f5922caf6bd.ali-cn-beijing.vectordb.zilliz.com.cn:19530",
        token="db_admin:Ad6^[1SXqXY(Zu~V"
    )
    model_zh = SentenceTransformer('/root/Data2/ZXN/llama3.1/qwen-llama/agent_evimed/bge-large-zh-v1.5')
    instruction_zh = "为这个句子生成表示以用于检索相关文章：" 
    q_embeddings_zh = model_zh.encode([instruction_zh + q for q in [content.strip()]], normalize_embeddings=True)

    #Returns:    List[List[dict]]: A nested list of dicts containing the result data. Embeddings are not included in the result data.
    prompt_res_zh = ziiiiz_client.search(
        collection_name='lingxi_rag',
        data=[q_embeddings_zh[0].tolist()],
        filter='data_type=="chinese document"',
        limit=5,
        output_fields=["id", "title", "question", "answer"]
    )

    model_en = SentenceTransformer('/root/Data2/ZXN/llama3.1/qwen-llama/agent_evimed/bge-large-en-v1.5')
    instruction_en = "Represent this sentence for searching relevant passages:"
    try:
        fanyi_result = baidu_fanyi(content.strip())
    except:
        fanyi_result = main_tmp_fanyi(
            "Translate the following text into English, require only the English translation results, do not output any explanatory text:\nThe given text is:\n" + content)
        # print(fanyi_result)
    q_embeddings_en = model_en.encode([instruction_en + e for e in [fanyi_result]], normalize_embeddings=True)    # 英文题目转的向量

    prompt_res_en = ziiiiz_client.search(
        collection_name='lingxi_rag',
        data=[q_embeddings_en[0].tolist()],
        filter='data_type=="english document"',
        limit=5,
        output_fields=["id", "title", "question", "answer"]
    )

    ids_tmp_zh = [i["id"] for i in prompt_res_zh[0]]
    ids_tmp_en = [i["id"] for i in prompt_res_en[0]]
    ids_tmp = ids_tmp_zh + ids_tmp_en
    ids = []
    for id in ids_tmp:
        if id.count("-") == 2:
            new_lis_id = id.rsplit("-", 1)
            ids.append(new_lis_id[0])
        else:
            ids.append(id)

    qc_id_tmp = []
    qc_id_lis = []
    for q in range(len(ids)):
        if ids[q] not in qc_id_tmp:
            qc_id_tmp.append(ids[q])
            qc_id_lis.append(ids_tmp[q])
    
    text_zh = [item['entity']['question'] + "\n" + item['entity']['answer'] for item in prompt_res_zh[0] if item["id"] in qc_id_lis]
    text_en = [item['entity']['question'] + "\n" + item['entity']['answer'] for item in prompt_res_en[0] if item["id"] in qc_id_lis]

    search_result_list_all = text_en + text_zh

    count_px = 1
    px_list = []
    for u in search_result_list_all:
        px_list.append("[" + str(count_px) + "] " + str(u))
        count_px += 1
    
    return px_list,qc_id_tmp


if __name__ == '__main__':

    start_time = time.time()
    mode = 'test'
    # mode = 'run'

    # user_input = '司美格鲁肽的说明书，临床效果，不良反应等信息' # user_input = '司美格鲁肽'
    # user_input = '司美格鲁肽'
    # user_input = '雷贝拉唑'
    # user_input = '脊柱侧弯的分类，什么程度的脊柱侧弯需要手术干预？'

    # user_input = '脊柱侧弯的临床诊断方式'
    user_input = '雷贝拉唑的临床效果'
    # user_input = '怎样治疗慢性肾衰？'
    print('初始问题：',user_input)

# try:
    #拆分是医学问题还是药学问题并输出其类型
    ques_type = ques_understanding_agent(user_input)
    flag_1_time = time.time()
    analyze_time = flag_1_time - start_time
    print('analyzetime',analyze_time)

    #answers 
    ques_type,answer = ques_answering_agent(user_input,ques_type)
    flag_2_time = time.time()
    answer_time = flag_2_time - flag_1_time
    print('answertime',answer_time)
    

    #review1  
    review1 = answer_reviewing_agent(user_input,answer)
    flag_3_time = time.time()
    review_time = flag_3_time - flag_2_time
    
    print('answer_review_time',review_time)

# except Exception as e:
#     if mode=='test':
#         print(str(e))
#         print('backup solution:')
#     query = '请从医学角度回答此问题，不要输出任何其他无关内容。在组织回答的格式时，使用总-分-总的结构呈现回答。先用一两句话简单介绍本问题主要研究对象，然后根据实际需要，分条回答详细内容；最后一段给出总结性回答' + '问题：' + user_input
#     messages = [{'role': 'system', 'content': '你是一个医学专家，有丰富的临床医学知识，擅长回答问题.'},
#             {'role': 'user', 'content': query}]
#     res = gpt4o_claude_func(messages,"gpt-4o-mini")
#     print(res)
