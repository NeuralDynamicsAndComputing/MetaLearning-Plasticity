import torch
from torch import nn, optim
import torch.nn.functional as F

from torchviz import make_dot


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cl1 = nn.Linear(25, 60)
        self.cl2 = nn.Linear(60, 84)
        self.fc1 = nn.Linear(84, 10)

        self.feed_back()

    def feed_back(self):
        self.cl1 = nn.Linear(25, 60)
        self.cl2 = nn.Linear(60, 84)
        self.fc3 = nn.Linear(84, 10)


        self.test_param_1 = nn.Parameter(torch.randn(60, 25))
        self.test_param_2 = nn.Parameter(torch.randn(84, 60))
        self.test_param_3 = nn.Parameter(torch.randn(84, 10))

        self.feed_back_params = nn.ParameterList()

        self.feed_back_params.append(self.test_param_1)
        self.feed_back_params.append(self.test_param_2)
        self.feed_back_params.append(self.test_param_3)

        self.feed_back_params_dict = nn.ParameterDict({
                'fdb_1': nn.Parameter(torch.randn(60, 25)),
                'fdb_2': nn.Parameter(torch.randn(84, 60)),
                'fdb_3': nn.Parameter(torch.randn(84, 10))
        })

        self.register_parameter(name='test_param', param=torch.nn.Parameter(torch.randn(84, 120)))

    def forward(self, x):
        x = F.relu(self.cl1(x))
        x = F.relu(self.cl2(x))
        x = F.log_softmax(self.fc1(x), dim=1)
        return x


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


model = MyModel()
model.fc1.register_forward_hook(get_activation('fc1'))
x = torch.randn(1, 25)

target = torch.empty(1, dtype=torch.long).random_(10)

optimizer = optim.Adam(model.feed_back_params, lr=1e-3)  # todo: pass only meta-params
loss_func = nn.CrossEntropyLoss()



for i in range(5):

    output = model(x)

    with torch.no_grad():
        for param, feed_back in zip(model.parameters(), model.feed_back_params):
            new_val = param + feed_back
            param.copy_(new_val)

    if i == 0:
        print(dict(list(model.named_parameters())))
        # print(dict(model.feed_back_params))
        print(model.feed_back_params_dict)

        make_dot(output, params=dict(list(model.named_parameters()))).render('model_torchviz', format='png')
        # make_dot(output, params=dict(list(model.parameters()))).render('model_torchviz', format='png')
        # make_dot(output, params=model.feed_back_params_dict).render('model_torchviz', format='png')

    # model.fc2.weight.data = model.fc2.weight + model.test_param_1

    # with torch.no_grad():
    #     model.fc2.weight = model.fc2.weight + model.test_param
    #
    # if i == 1:
    #     make_dot(output, params=dict(list(model.named_parameters()))).render('model_torchviz', format='png')
    #
    #     for i in model.named_parameters():
    #         print(i[0])
    #
    #     for i in model.feed_back_params:
    #         print(i[0])
    #
    # # with torch.no_grad():
    # #     new_param = model.fc2.weight + model.test_param
    # #     model.fc2.weight.copy_(new_param)


    # print(activation['fc2'])

    loss = loss_func(output, target)

    # print()

    optimizer.zero_grad()
    loss.backward()#inputs=model.feed_back_params)
    optimizer.step()



    # maml.net.plasticity = nn.ParameterList()