def aa(a, b, /):
    a, b = 1, 1
    return a, b

a, b = aa(1, 2)
print(a, b)


a = [1.1231232345254, 2.35345245345, 3.23523452]
a_rounded = [round(num, 2) for num in a]
print(a_rounded)

print([1,2,3]*[4,5,6])