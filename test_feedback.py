import torch
from torch import nn, optim
import torch.nn.functional as F
from Optim_rule import MyOptimizer
from torchviz import make_dot


torch.manual_seed(0)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cl1 = nn.Linear(25, 60)
        self.cl2 = nn.Linear(60, 84)
        self.fc1 = nn.Linear(84, 10)

        self.feed_back()

    def feed_back(self):

        # self.forward_params_list = model.named_parameters()

        self.feed_back_params_list = nn.ParameterList([
            nn.Parameter(torch.randn(60, 25)),
            nn.Parameter(torch.randn(84, 60)),
            nn.Parameter(torch.randn(84, 10))
        ])

        self.feed_fwd_params_list = nn.ParameterList([
            self.cl1.weight,
            self.cl1.bias,
            self.cl2.weight,
            self.cl2.bias,
            self.fc1.weight,
            self.fc1.bias
        ])


        #
        # self.feed_back_params_dict = nn.ParameterDict({
        #     'fdb_1': nn.Parameter(torch.randn(60, 25)),
        #     'fdb_2': nn.Parameter(torch.randn(84, 60)),
        #     'fdb_3': nn.Parameter(torch.randn(84, 10))
        # })

        # self.register_parameter(name='test_param', param=torch.nn.Parameter(torch.randn(84, 120)))

    def forward(self, x):
        x = F.relu(self.cl1(x))
        x = F.relu(self.cl2(x))

        return self.fc1(x)

def main():
    # -- register hook
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # -- data
    x = torch.randn(1, 25)
    target = torch.empty(1, dtype=torch.long).random_(10)

    # -- model
    model = MyModel()
    SGD = True
    # model.fc1.register_forward_hook(get_activation('fc1'))
    # optim_meta = optim.Adam(model.feed_back_params, lr=1e-3)  # todo: pass only meta-params
    # optim_meta = optim.SGD(model.feed_back_params_list, lr=1e-3)  # todo: pass only meta-params
    if SGD:
        optimizer = optim.SGD(model.parameters(), lr=1e-3)  # todo: pass only meta-params
    else:
        optimizer = MyOptimizer(model.feed_fwd_params_list, lr=1e-3)  # todo: pass only meta-params
    loss_func = nn.CrossEntropyLoss()

    # -- train
    for i in range(500):

        # -- predict
        output = model(x)

        # -- loss
        loss = loss_func(output, target)
        print('Epoch {}, loss: {}'.format(i, loss.item()))

        # -- optimize
        if SGD:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            optimizer.step(loss)

        # # -- update params
        # with torch.no_grad():
        #     for param, feed_back in zip(model.parameters(), model.feed_back_params):
        #         new_val = param + feed_back
        #         param.copy_(new_val)
        #
        # # -- draw graph
        # if i == 0:
        #     make_dot(output, params=dict(list(model.named_parameters()))).render('model_torchviz', format='png')

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

        # loss = loss_func(output, target)
        #
        # # print()
        #
        # optim_meta.zero_grad()
        # loss.backward()#inputs=model.feed_back_params)
        # optim_meta.step()
        #
        #
        #
        # # maml.net.plasticity = nn.ParameterList()

main()