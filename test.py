import dynet as dy

model = dy.Model()
input_lookup = model.add_lookup_parameters((100, 128))

pass