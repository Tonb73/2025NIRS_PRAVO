import pandas as pd
import random

# Список категорий, соответствующих проблемам с правами
categories = [
    "Интернет-мошенничество", "Трудовые споры", "Защита прав потребителей",
    "Жилищные споры", "Семейные споры", "Нарушение прав человека",
    "Незаконное увольнение", "Дискриминация", "Проблемы с банками",
    "Нарушение авторских прав", "Проблемы с соцобеспечением",
    "Нарушение прав собственности", "Проблемы с налогами",
    "Проблемы с образованием", "Проблемы с медицинским обслуживанием"
]

# Шаблоны для описания проблем. Для уникальности добавляется номер случая.
templates = [
    "Обнаружил(а) нарушение – {detail}. Требую защиты, так как {issue}.",
    "Мною была выявлена проблема: {detail}. Это ущемляет мои права, потому что {issue}.",
    "Столкнулся(лась) с ситуацией: {detail}. Мои законные права нарушаются, так как {issue}.",
    "Имеется нарушение – {detail}. Считаю, что {issue} требует вмешательства.",
    "Возникла проблема: {detail}. Необходима помощь, ведь {issue}.",
    "Нарушение обнаружено – {detail}. Это негативно сказывается на моих правах, так как {issue}.",
    "Я столкнулся(лась) с проблемой: {detail}. Это не допустимо, потому что {issue}.",
    "Выявлено нарушение: {detail}. Считаю, что {issue} требует немедленного разрешения."
]

# Дополнительные детали, которые помогут разнообразить описания
details = [
    "отказ в возврате предоплаты за онлайн-курсы",
    "несвоевременная выплата заработной платы",
    "продажа товара с заведомо известными дефектами",
    "необоснованный отказ в предоставлении услуг связи",
    "задержка оплаты коммунальных услуг без объяснения причин",
    "неправомерное изменение условий договора аренды",
    "недостоверная информация о качестве товара",
    "несоблюдение условий трудового договора",
    "неправомерное начисление штрафов за несуществующие нарушения",
    "ограничение доступа к информации о правах потребителя",
    "отказ банка в удовлетворении заявки на кредит",
    "необоснованное повышение тарифов на связь",
    "несанкционированное использование личных данных",
    "отказ в выплате страхового возмещения",
    "ограничение прав при оформлении социальных пособий",
    "недобросовестное отношение к сотрудникам компании",
    "несвоевременное предоставление медицинской помощи",
    "нарушение авторских прав при копировании контента",
    "неправомерное увольнение без объяснения причин",
    "ограничение возможностей для получения образования"
]

# Дополнительные причины, почему это нарушение прав
issues = [
    "это ставит под угрозу моё финансовое благополучие",
    "из-за этого я испытываю значительные материальные потери",
    "это нарушает мои законные ожидания",
    "такая ситуация негативно сказывается на качестве жизни",
    "это препятствует нормальной работе и личному развитию",
    "это вызывает постоянный стресс и беспокойство",
    "я лишён(а) возможности реализовать свои права",
    "это влияет на моё здоровье и безопасность",
    "это ставит под угрозу благополучие моей семьи",
    "из-за этого я не могу получить обещанные услуги",
    "это нарушает принципы справедливости и равенства",
    "такое обращение недопустимо с точки зрения закона",
    "это приводит к серьезным финансовым проблемам",
    "такое нарушение требует немедленной правовой защиты",
    "это ограничивает мои возможности для развития"
]

# Список для хранения записей
records = []

# Генерируем 1500 уникальных записей
for i in range(1, 1501):
    cat = random.choice(categories)
    template = random.choice(templates)
    detail = random.choice(details)
    issue = random.choice(issues)
    desc = template.format(num=i, detail=detail, issue=issue)
    records.append({"Desc": desc, "Cat": cat})

# Создаем DataFrame и сохраняем его в Excel-файл
df_filled = pd.DataFrame(records)
output_file = "Learn_filled_unique.xlsx"
df_filled.to_excel(output_file, index=False)

print(f"Файл успешно сохранен как {output_file}")
