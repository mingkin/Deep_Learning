# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2019/3/29 0029 下午 3:21
"""



from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


'source: https://habr.com/en/post/439768/'

'''首先，让我们加载图像，拉伸它的宽度，从中间裁剪一条水平线，将其转换为b/w颜色并保存为数组。'''

image_path = "barcode.jpeg"
img = Image.open(image_path)
width, height = img.size
basewidth = 4*width
img = img.resize((basewidth, height), Image.ANTIALIAS)
hor_line_bw = img.crop((0, int(height/2), basewidth, int(height/2) + 1)).convert('L')
hor_data = np.asarray(hor_line_bw, dtype="int32")[0]

print('hor_data:', hor_data)
'''条形码上的黑线对应于«1»，但是在RGB中黑线是相反的，0，所以数组需要倒排。我们还将计算平均值。'''
hor_data = 255 - hor_data
avg = np.average(hor_data)

# plt.plot(hor_data)
# plt.show()

print('hor_data:', hor_data)
print(avg)
'现在我们需要确定一个“位”的宽度。为此，我们将提取序列，保存平均线交叉的位置。'

pos1, pos2 = -1, -1
bits = ""
for p in range(basewidth - 2):
    if hor_data[p] < avg and hor_data[p + 1] > avg:
        bits += "1"
        if pos1 == -1:
            pos1 = p
        if bits == "101":
            pos2 = p
            break
    if hor_data[p] > avg and hor_data[p + 1] < avg:
        bits += "0"

bit_width = int((pos2 - pos1)/3)
print('bits,bit_width, pos1,pos2', bits, bit_width, pos1, pos2)


'现在让我们自己解码。我们需要找到每一个平均线交叉，并找到最后一个区间的比特数。'
'数字将不匹配完美(代码可以拉伸或弯曲一点)，因此，我们需要将值四舍五入为integer。'

bits = ""
for p in range(basewidth - 2):
    if hor_data[p] > avg and hor_data[p + 1] < avg:
        interval = p - pos1
        cnt = interval/bit_width
        bits += "1"*int(round(cnt))
        pos1 = p
    if hor_data[p] < avg and hor_data[p + 1] > avg:
        interval = p - pos1
        cnt = interval/bit_width
        bits += "0"*int(round(cnt))
        pos1 = p

print('pos1, pos2, bits', pos1, pos2,bits)





'''
在我们的例子中，序列的开头是11010010000，它对应于«Code B»。
我懒得手动输入所有代码，所以我只是从Wikipedia页面复制粘贴了它。
这几行代码的解析也是在Python上进行的(提示——不要在生产环境中做这样的事情)。
'''

CODE128_CHART = """
        0	_	_	00	32	S	11011001100	212222
        1	!	!	01	33	!	11001101100	222122
        2	"	"	02	34	"	11001100110	222221
        3	#	#	03	35	#	10010011000	121223
        4	$	$	04	36	$	10010001100	121322
        5	%	%	05	37	%	10001001100	131222
        6	&	&	06	38	&	10011001000	122213
        7	'	'	07	39	'	10011000100	122312
        8	(	(	08	40	(	10001100100	132212
        9	)	)	09	41	)	11001001000	221213
        10	*	*	10	42	*	11001000100	221312
        11	+	+	11	43	+	11000100100	231212
        12	,	,	12	44	,	10110011100	112232
        13	-	-	13	45	-	10011011100	122132
        14	.	.	14	46	.	10011001110	122231
        15	/	/	15	47	/	10111001100	113222
        16	0	0	16	48	0	10011101100	123122
        17	1	1	17	49	1	10011100110	123221
        18	2	2	18	50	2	11001110010	223211
        19	3	3	19	51	3	11001011100	221132
        20	4	4	20	52	4	11001001110	221231
        21	5	5	21	53	5	11011100100	213212
        22	6	6	22	54	6	11001110100	223112
        23	7	7	23	55	7	11101101110	312131
        24	8	8	24	56	8	11101001100	311222
        25	9	9	25	57	9	11100101100	321122
        26	:	:	26	58	:	11100100110	321221
        27	;	;	27	59	;	11101100100	312212
        28	<	<	28	60	<	11100110100	322112
        29	=	=	29	61	=	11100110010	322211
        30	>	>	30	62	>	11011011000	212123
        31	?	?	31	63	?	11011000110	212321
        32	@	@	32	64	@	11000110110	232121
        33	A	A	33	65	A	10100011000	111323
        34	B	B	34	66	B	10001011000	131123
        35	C	C	35	67	C	10001000110	131321
        36	D	D	36	68	D	10110001000	112313
        37	E	E	37	69	E	10001101000	132113
        38	F	F	38	70	F	10001100010	132311
        39	G	G	39	71	G	11010001000	211313
        40	H	H	40	72	H	11000101000	231113
        41	I	I	41	73	I	11000100010	231311
        42	J	J	42	74	J	10110111000	112133
        43	K	K	43	75	K	10110001110	112331
        44	L	L	44	76	L	10001101110	132131
        45	M	M	45	77	M	10111011000	113123
        46	N	N	46	78	N	10111000110	113321
        47	O	O	47	79	O	10001110110	133121
        48	P	P	48	80	P	11101110110	313121
        49	Q	Q	49	81	Q	11010001110	211331
        50	R	R	50	82	R	11000101110	231131
        51	S	S	51	83	S	11011101000	213113
        52	T	T	52	84	T	11011100010	213311
        53	U	U	53	85	U	11011101110	213131
        54	V	V	54	86	V	11101011000	311123
        55	W	W	55	87	W	11101000110	311321
        56	X	X	56	88	X	11100010110	331121
        57	Y	Y	57	89	Y	11101101000	312113
        58	Z	Z	58	90	Z	11101100010	312311
        59	[	[	59	91	[	11100011010	332111
        60	\	\	60	92	\	11101111010	314111
        61	]	]	61	93	]	11001000010	221411
        62	^	^	62	94	^	11110001010	431111
        63	_	_	63	95	_	10100110000	111224
        64	NUL	`	64	96	`	10100001100	111422
        65	SOH	a	65	97	a	10010110000	121124
        66	STX	b	66	98	b	10010000110	121421
        67	ETX	c	67	99	c	10000101100	141122
        68	EOT	d	68	100	d	10000100110	141221
        69	ENQ	e	69	101	e	10110010000	112214
        70	ACK	f	70	102	f	10110000100	112412
        71	BEL	g	71	103	g	10011010000	122114
        72	BS	h	72	104	h	10011000010	122411
        73	HT	i	73	105	i	10000110100	142112
        74	LF	j	74	106	j	10000110010	142211
        75	VT	k	75	107	k	11000010010	241211
        76	FF	l	76	108	l	11001010000	221114
        77	CR	m	77	109	m	11110111010	413111
        78	SO	n	78	110	n	11000010100	241112
        79	SI	o	79	111	o	10001111010	134111
        80	DLE	p	80	112	p	10100111100	111242
        81	DC1	q	81	113	q	10010111100	121142
        82	DC2	r	82	114	r	10010011110	121241
        83	DC3	s	83	115	s	10111100100	114212
        84	DC4	t	84	116	t	10011110100	124112
        85	NAK	u	85	117	u	10011110010	124211
        86	SYN	v	86	118	v	11110100100	411212
        87	ETB	w	87	119	w	11110010100	421112
        88	CAN	x	88	120	x	11110010010	421211
        89	EM	y	89	121	y	11011011110	212141
        90	SUB	z	90	122	z	11011110110	214121
        91	ESC	{	91	123	{	11110110110	412121
        92	FS	|	92	124	|	10101111000	111143
        93	GS	}	93	125	}	10100011110	111341
        94	RS	~	94	126	~	10001011110	131141
        103	Start Start A	208	SCA	11010000100	211412
        104	Start Start B	209	SCB	11010010000	211214
        105	Start Start C	210	SCC	11010011100	211232
        106	Stop Stop	-	- -	11000111010	233111""".split()
SYMBOLS = [value for value in CODE128_CHART[6::8]]
VALUESB = [value for value in CODE128_CHART[2::8]]
CODE128B = dict(zip(SYMBOLS, VALUESB))
print('SYMBOLS',SYMBOLS)

'最后一部分很简单。首先，我们将序列分割为11位块:'
sym_len = 11
symbols = [bits[i:i+sym_len] for i in range(0, len(bits), sym_len)]


print('symbols', symbols)
'最后，生成输出字符串并显示:'


str_out = ""
for sym in symbols:
    print(sym)
    try:
        if CODE128B[sym] == 'Start':
            continue
        elif CODE128B[sym] == 'Stop':
            break
        str_out += CODE128B[sym]
        print("  ", sym, CODE128B[sym])
    except:
        str_out += '*'
        print("  *")

print("Str:", str_out)








'pip install pyzbar '

from pyzbar.pyzbar import decode

img = Image.open(image_path)
decode = decode(img)
print(decode)






