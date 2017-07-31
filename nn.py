import miniflow as mf

x = mf.Input()
y = mf.Input()

f = mf.Add(x, y)

feed_dict = {x: 5, y: 10}

sorted_nodes = mf.topological_sort(feed_dict)
print(mf.forward_pass(f, sorted_nodes))
