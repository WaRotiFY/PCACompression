from untiles import ConImage
import sys

argv = sys.argv[1:]
if (argv[0][-4:] == ".rdc"):
    try:
        ConImage.reverse_codek_img(argv[1], argv[0])
    except:
        print("Second arg error")
else:
    codek_img = ConImage(argv[0], float(argv[1]))
    codek_img.save_wb_img()
    codek_img.make_bin_rdc(argv[2] + str(".rdc"))
    codek_img.reverse_codek_img("dump" + str(".jpg"))