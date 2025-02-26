# #程序练习题2.2
# Money=input("请输入带有标识的金钱值：")
# if Money[-1] in ['r','R']:
#     D=eval(Money[0:-1])/6
#     print("转换后的美元为{:23.2f}".format(D))
# elif Money[-1] in ['d','D']:
#     R=eval(Money[0:-1])*6
#     print("转换后的人民币为{:.2f}".format(R))
# else:
#     print("输入格式错误")
#
#
#
# x=input()
# if x==x[::-1]:
#     print("yes")
# else:
#     print("no")

a = ['f','s',3,3,4,2,'d',4,5,6,1]
b = set(a)
a = list(set(a))
print(a)
