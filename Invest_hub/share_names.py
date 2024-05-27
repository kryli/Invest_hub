import os

def process_files_in_folder(folder_path, exclude_file):
    for filename in os.listdir(folder_path):
        if filename == exclude_file or not filename.endswith('.txt'):
            continue

        file_path = os.path.join(folder_path, filename)
        process_file(file_path)

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        ticker, figi = line.strip().split(', ')[:2]
        company_name = get_company_name(ticker)
        updated_line = f"{ticker}, {figi}, {company_name}\n"
        updated_lines.append(updated_line)

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

def get_company_name(ticker):
    company_names = {
    'MGNT': 'Магнит',
    'MVID': 'М.Видео',
    'AFLT': 'Аэрофлот',
    'BELU': 'НоваБев Групп',
    'AQUA': 'Инарктика',
    'APTK': 'Аптека 36.6',
    'GCHE': 'Черкизово',
    'DSKY': 'Детский мир',
    'SVAV': 'Соллерс',
    'HNFG': 'HENDERSON',
    'ABRD': 'Абрау-Дюрсо',
    'LENT': 'Лента',
    'UPRO': 'Юнипро',
    'IRAO': 'Интер РАО',
    'FEES': 'ФСК ЕЭС',
    'HYDR': 'РусГидро',
    'MSNG': 'Мосэнерго',
    'TGKA': 'ТГК-1',
    'OGKB': 'ОГК-2',
    'MSRS': 'МРСК Московский Регион',
    'MRKP': 'МРСК Центра и Приволжья',
    'TGKB': 'ТГК-2',
    'MRKC': 'МРСК Центра',
    'MRKV': 'МРСК Волги',
    'MRKZ': 'МРСК Северо-Запада',
    'TGKN': 'ТГК-11',
    'LSNG': 'Ленэнерго',
    'LSNGP': 'Ленэнерго (прив)',
    'MRKY': 'МРСК Юга',
    'MRKS': 'МРСК Сибири',
    'KLSB': 'Калугаэнергосбыт',
    'MRKU': 'МРСК Урала',
    'DVEC': 'ДЭК',
    'VRSB': 'Воронежская сбытовая компания',
    'TGKBP': 'ТГК-2 (прив)',
    'PMSB': 'Пермэнергосбыт',
    'PMSBP': 'Пермэнергосбыт (прив)',
    'LKOH': 'Лукойл',
    'GAZP': 'Газпром',
    'TRNFP': 'Транснефть (прив)',
    'SNGSP': 'Сургутнефтегаз (прив)',
    'NVTK': 'Новатэк',
    'ROSN': 'Роснефть',
    'TATN': 'Татнефть',
    'SNGS': 'Сургутнефтегаз',
    'RNFT': 'Руснефть',
    'BANEP': 'Башнефть (прив)',
    'SIBN': 'Газпром нефть',
    'TATNP': 'Татнефть (прив)',
    'BANE': 'Башнефть',
    'ELFV': 'ЭЛ5-Энерго',
    'KRKNP': 'Саратовский НПЗ (прив)',
    'YAKG': 'ЯТЭК',
    'TCSG': 'ТКС Холдинг',
    'SBER': 'Сбербанк',
    'SFIN': 'ЭсЭфАй',
    'VTBR': 'ВТБ',
    'MOEX': 'Московская биржа',
    'AFKS': 'АФК Система',
    'SVCB': 'Совкомбанк',
    'SBERP': 'Сбербанк (прив)',
    'BSPB': 'Банк Санкт-Петербург',
    'RENI': 'Ренессанс Страхование',
    'CBOM': 'МКБ',
    'CARM': 'КарМани',
    'MGKL': 'Мосгорломбард',
    'LIFE': 'Фармсинтез',
    'GECO': 'Генетико',
    'ABIO': 'Артген',
    'GEMC': 'UMC',
    'LSRG': 'ЛСР Групп',
    'PIKK': 'ПИК',
    'SMLT': 'Самолет',
    'POSI': 'Positive Tech',
    'ASTR': 'Группа Астра',
    'VKCO': 'ВК',
    'DELI': 'Делимобиль',
    'WUSH': 'Whoosh',
    'EUTR': 'ЕвроТранс',
    'FLOT': 'Совкомфлот',
    'NMTP': 'НМТП',
    'GTRK': 'ГТМ',
    'IRKT': 'Яковлев',
    'FESH': 'ДВМП',
    'UNAC': 'ОАК',
    'KMAZ': 'КАМАЗ',
    'UWGN': 'ОВК',
    'NKHP': 'НКХП',
    'MSTT': 'Мостотрест',
    'RKKE': 'РКК',
    'MTLR': 'Мечел',
    'GMKN': 'Норильский никель',
    'CHMF': 'Северсталь',
    'PLZL': 'Полюс',
    'NLMK': 'НЛМК',
    'ALRS': 'Алроса',
    'MAGN': 'ММК',
    'MTLRP': 'Мечел (прив)',
    'RASP': 'Распадская',
    'TRMK': 'Трубная металлургическая компания',
    'UGLD': 'Южуралзолото ГК',
    'RUAL': 'Русал',
    'PHOR': 'ФосАгро',
    'SGZH': 'Сегежа',
    'SELG': 'Селигдар',
    'ROLO': 'Русолово',
    'ENPG': 'Эн+',
    'VSMO': 'ВСМПО-АВИСМА',
    'BLNG': 'Белон',
    'LNZL': 'Лензолото',
    'AMEZ': 'Ашинский метзавод',
    'LNZLP': 'Лензолото (прив)',
    'AKRN': 'Акрон',
    'KZOS': 'Казаньоргсинтез',
    'PRFN': 'ЧЗПСН',
    'NKNCP': 'Нижнекамскнефтехим (прив)',
    'KZOSP': 'Казаньоргсинтез (прив)',
    'NKNC': 'Нижнекамскнефтехим',
    'KAZT': 'КуйбашевАзот',
    'CHMK': 'ЧМК',
    'KAZTP': 'КуйбашевАзот (прив)',
    'UNKL': 'ЮУНК',
    'RTKM': 'Ростелеком',
    'MTSS': 'МТС',
    'RTKMP': 'Ростелеком (прив)',
    'NSVZ': 'Наука-Связь',
    'CNTL': 'Центральный Телеграф',
    'TTLK': 'Таттелеком',
    'MGTSP': 'МГТС (прив)',
    'CNTLP': 'Центральный Телеграф (прив)',
    'OZON' : 'Ozon Holdings PLC',
    'AGRO' : 'РусАгро',
    'KROT' : 'Красный Октябрь',
    'OKEY' : 'O.Key Group SA',
    'ETLN' : 'Etalon Group PLC ГДР'

    }
    return company_names.get(ticker, 'Unknown')

data_folder = 'data/data_for_tikers_by_spheres'
exclude_file = 'materials.txt'
process_files_in_folder(data_folder, exclude_file)