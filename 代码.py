#!/usr/bin/env python
# coding: utf-8

# # **1、准备数据:**
# 
#     创建数据集和数据字典
# 
#     创建数据读取器train_reader 和test_reader
# 
# # **2、配置网络**
# 
# 定义网络
# 
# 定义损失函数
# 
# 定义优化算法
# 
# # **3、训练网络**
# 
# # **4、模型评估**
# 
# # **5、模型预测**
# 

# In[ ]:


get_ipython().system('pip install Beautifulsoup4 ')


# In[ ]:


# 导入必要的包
import os
from multiprocessing import cpu_count
import numpy as np
import shutil
import paddle
import paddle.fluid as fluid



# In[53]:


# 创建数据集和数据字典

data_root_path='/home/aistudio/data/'

def create_data_list(data_root_path):
    with open(data_root_path + 'test_list.txt', 'w') as f:
        pass
    with open(data_root_path + 'train_list.txt', 'w') as f:
        pass

    with open(os.path.join(data_root_path, 'dict_txt.txt'), 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])

    with open(os.path.join(data_root_path, 'news_classify_data.txt'), 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()
    i = 0
    for line in lines:
        title = line.split('_!_')[-1].replace('\n', '')
        l = line.split('_!_')[1]
        labs = ""
        if i % 10 == 0:
            with open(os.path.join(data_root_path, 'test_list.txt'), 'a', encoding='utf-8') as f_test:
                for s in title:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + l + '\n'
                f_test.write(labs)
        else:
            with open(os.path.join(data_root_path, 'train_list.txt'), 'a', encoding='utf-8') as f_train:
                for s in title:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + l + '\n'
                f_train.write(labs)
        i += 1
    print("数据列表生成完成！")


# 把下载得数据生成一个字典
def create_dict(data_path, dict_path):
    dict_set = set()
    # 读取已经下载得数据
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 把数据生成一个元组
    for line in lines:
        title = line.split('_!_')[-1].replace('\n', '')
        for s in title:
            dict_set.add(s)
    # 把元组转换成字典，一个字对应一个数字
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])
        i += 1
    # 添加未知字符
    dict_txt = dict(dict_list)
    end_dict = {"<unk>": i}
    dict_txt.update(end_dict)
    # 把这些字典保存到本地中
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))

    print("数据字典生成完成！")


# 获取字典的长度
def get_dict_len(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        line = eval(f.readlines()[0])

    return len(line.keys())


if __name__ == '__main__':
    # 把生产的数据列表都放在自己的总类别文件夹中
    data_root_path = "/home/aistudio/data/"
    data_path = os.path.join(data_root_path, 'news_classify_data.txt')
    dict_path = os.path.join(data_root_path, "dict_txt.txt")
    # 创建数据字典
    create_dict(data_path, dict_path)
    # 创建数据列表
    create_data_list(data_root_path)


# In[47]:


# 创建数据读取器train_reader 和test_reader
# 训练/测试数据的预处理
def data_mapper(sample):
    data, label = sample
    data = [int(data) for data in data.split(',')]
    return data, int(label)

# 创建数据读取器train_reader
def train_reader(train_list_path):
    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            # 打乱数据
            np.random.shuffle(lines)
            # 开始获取每张图像和标签
            for line in lines:
                data, label = line.split('\t')
                yield data, label
    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)
#  创建数据读取器test_reader
def test_reader(test_list_path):

    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data, label = line.split('\t')
                yield data, label

    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)


# # 卷积神经网络（Convolutional Neural Networks, CNN）
# 
# 输入词向量序列，产生一个特征图（feature map），对特征图采用时间维度上的最大池化（max pooling over time）操作得到此卷积核对应的整句话的特征，最后，将所有卷积核得到的特征拼接起来即为文本的定长向量表示，对于文本分类问题，将其连接至softmax即构建出完整的模型。
# 
# 在实际应用中，我们会使用多个卷积核来处理句子，窗口大小相同的卷积核堆叠起来形成一个矩阵，这样可以更高效的完成运算。
# 
# 另外，我们也可使用窗口大小不同的卷积核来处理句子.
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/3766261f24b54514b6cbc0d30270c6a3f38c1d0aaf8f450c97e8303eca51f204)

# In[48]:


# 创建CNN网络

def CNN_net(data,dict_dim, class_dim=10, emb_dim=128, hid_dim=128,hid_dim2=98):
        emb = fluid.layers.embedding(input=data,
                                 size=[dict_dim, emb_dim])
        conv_3 = fluid.nets.sequence_conv_pool(
                                                 input=emb,
                                                 num_filters=hid_dim,
                                                 filter_size=3,
                                                 act="tanh",
                                                 pool_type="sqrt")
        conv_4 = fluid.nets.sequence_conv_pool(
                                                 input=emb,
                                                 num_filters=hid_dim2,
                                                 filter_size=4,
                                                 act="tanh",
                                                 pool_type="sqrt")
                                                 
        output = fluid.layers.fc(
            input=[conv_3, conv_4], size=class_dim, act='softmax')
        return output


# In[49]:


# 定义输入数据， lod_level不为0指定输入数据为序列数据
words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
# 获取数据字典长度
dict_dim = get_dict_len('/home/aistudio/data/dict_txt.txt')
# 获取卷积神经网络
# model = CNN_net(words, dict_dim, 15)
# 获取分类器
model = CNN_net(words, dict_dim)
# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取预测程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.002)
opt = optimizer.minimize(avg_cost)

# 创建一个执行器，CPU训练速度比较慢
place = fluid.CPUPlace()
# place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())


# In[50]:


# 获取训练数据读取器和测试数据读取器
train_reader = paddle.batch(reader=train_reader('/home/aistudio/data/train_list.txt'), batch_size=128)
test_reader = paddle.batch(reader=test_reader('/home/aistudio/data/test_list.txt'), batch_size=128)


# In[51]:


# 定义数据映射器
feeder = fluid.DataFeeder(place=place, feed_list=[words, label])


# In[52]:


EPOCH_NUM=10
model_save_dir = '/home/aistudio/work/infer_model/'
# 开始训练

for pass_id in range(EPOCH_NUM):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost, acc])

        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Acc:%0.5f' % (pass_id, batch_id, train_cost[0], train_acc[0]))
    # 进行测试
    test_costs = []
    test_accs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                              feed=feeder.feed(data),
                                              fetch_list=[avg_cost, acc])
        test_costs.append(test_cost[0])
        test_accs.append(test_acc[0])
    # 计算平均预测损失在和准确率
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))

# 保存预测模型
if not os.path.exists(model_save_dir): 
    os.makedirs(model_save_dir) 
fluid.io.save_inference_model(model_save_dir, 
                            feeded_var_names=[words.name], 
                            target_vars=[model], 
                            executor=exe)
print('训练模型保存完成！') 


# In[55]:



from bs4 import BeautifulSoup
from urllib import request
if __name__ == '__main__':
   target_url = 'https://news.so.com/hotnews?src=onebox'
   req = request.Request(target_url)
   response = request.urlopen(req)
   response = response.read().decode('utf8')
   soup = BeautifulSoup(response)
   y = 0
   data_tem =[]
   for tag in soup.find_all('a',class_='item'):
       for b in tag.find_all('span',class_='title'):
           data_tem.append(b.string)
   print(data_tem)


# In[56]:


# 用训练好的模型进行预测并输出预测结果
# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

save_path = '/home/aistudio/work/infer_model/'

# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)


# 获取数据
def get_data(sentence):
    # 读取数据字典
    with open('/home/aistudio/data/dict_txt.txt', 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    dict_txt = dict(dict_txt)
    # 把字符串数据转换成列表数据
    keys = dict_txt.keys()
    data = []
    for s in sentence:
        # 判断是否存在未知字符
        if not s in keys:
            s = '<unk>'
        data.append(int(dict_txt[s]))
    return data


data = []
# 获取图片数据
for item in data_tem:
    data.append(get_data(item))

# 获取每句话的单词数量
base_shape = [[len(c) for c in data]]

# 生成预测数据
tensor_words = fluid.create_lod_tensor(data, base_shape, place)

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: tensor_words},
                 fetch_list=target_var)

# 分类名称
names = [ '文化', '娱乐', '体育', '财经','房产', '汽车', '教育', '科技', '国际', '证券']

# 获取结果概率最大的label
for i in range(len(data)):
    lab = np.argsort(result)[0][i][-1]
    print('预测结果标签为：%d， 名称为：%s， 概率为：%f' % (lab, names[lab], result[0][i][lab]))


# In[ ]:




