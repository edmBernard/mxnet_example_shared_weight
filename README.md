# MXNet example: Shared Weight

Some test to partially shared layer between network

I made this repository to try different way to shared weight in different configuration : 
- siamese network
- common parts (We want a model able to predict Age and Gender but the database don't contain both attributs)

In 2017, the gluon API was still new that's why I try to use the Module API. Today, I advice to use the gluon API, it's more flexible and don't require hack to achieve shared weight.

```
.
├── demo_shared_partial_with_super_symbol_v1.py  not working: Example with Shared_module argument from mx.mod.bind
├── demo_shared_partial_with_super_symbol_v2.py  not working: Example with Shared_module argument from mx.mod.bind
├── demo_shared_with_weight_variable.py          not working: Ewample with shared weight
├── demo_with_gluon_one_network.py               Example with Gluon without shared weight
├── demo_shared_full_net.py                      Example with Shared_module argument from mx.mod.bind
├── demo_shared_with_weight_copy_each_time.py    Example with weight transfert each time
├── demo_with_gluon.py                           Example with Sequential gluon API
├── demo_with_gluon_hybrid.py                    Example with HybridSequential gluon API
├── demo_with_gluon_inside_param.py              Example with condition in Block forward definition
├── demo_with_gluon_siamese.py                   Example of siamese network with gluon
└── README.md
```
