from src.imports import *
from src.navqt import NAVQT


def run():
    try:
        args = sys.argv[-1].split("|")
    except:
        args = []
    print("ARGUMENTS")
    print(args)
    print("---------")
    qc = NAVQT()
    kwargs = {"savepth": "./results/"}
    for _, arg in enumerate(args):
        try:
            var = arg.split("=")[0]

            if type(getattr(qc, var)) is bool:
                val = arg.split("=")[1].lower() == "true"
            elif type(getattr(qc, var)) is int:
                val = int(arg.split("=")[1])
            elif type(getattr(qc, var)) is float:
                val = float(arg.split("=")[1])
            elif type(getattr(qc, var)) is str:
                val = arg.split("=")[1]
            else:
                val = None
                print("COULD NOT FIND VARIABLE:", var)
            kwargs.update({var: val})
            print(var, ":", val)
        except:
            if "main.py" not in arg:
                print("Trouble with " + arg)
            pass

    qc = NAVQT(**kwargs)
    if not os.path.isfile(qc.savepth + "history---" + qc.settings + ".pdf"):
        print(qc)
        qc.train(n_epochs=qc.max_iter, early_stop=True, grad_norm=True)
        qc.plot_history(save=True)
        print(
            "Succesfully saved file(s) to:",
            qc.savepth + "history---" + qc.settings + ".*",
        )
    else:
        print("File exists!")


if __name__ == "__main__":
    run()
