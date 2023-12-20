hito_exe = int(input("Introducir hito que se desea ejecutar: "))

if hito_exe == 1:
    import Archivos_hitos.Hito_1
elif hito_exe == 2:
    import Archivos_hitos.Hito_2
elif hito_exe == 3:
    import Archivos_hitos.Hito_3
elif hito_exe == 4:
    import Archivos_hitos.Hito_4
elif hito_exe == 5:
    import Archivos_hitos.Hito_5
elif hito_exe == 6:
    import Archivos_hitos.Hito_6
# elif hito_exe == 7:
    # import Hitos.Hito_7
else:
    print('Numero de hito incorrecto')
    exit