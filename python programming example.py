'''
# string practice 1
str1 = "python is interpreter language\n"
str2 = '"python is very easy language to learn"'
print(str1, str2)
str1 = "We have to lean the \"python\""
str2 = """What happen if we use the 3 double single quotation"""
print(str1, str2)

# string practice 2
multiline = "Life is too short\nYou need python"
print(multiline)


#string Operation
str1 = "Hi Everyone"
str2 = "My name is james"
str3 = "**********" # Comment Block
print(f'{str3*10}\n{str1+str2},\n'
      f'This Block is comment block\n'
      f'{str3*10} ')

# indexing / slicing
lenStr2 = len(str2)
# j --> "J"
print(str2[-5])
print(str2)

str2 = "20210308Sunny"
print('Today weather is ' + '\"' + str2[8:] + '\"')

list_a = [1, 2, 3, 4, 5]
print(list_a[1])
list_a[1] = 3
print(list_a)
'''
str2 = "My name is james shin"
# str2[-4] = 'S'
str2 = "My name is james shin"
print(len(str2))
str_a = str2[:-4]
str_b = "S"
str_c = str2[18:21]
str_full = str_a + str_b + str_c
print(str_a)
print(str_b)
print(str_c)
print(str_full)

str1 = "abcdef"
print(str1[2:4])