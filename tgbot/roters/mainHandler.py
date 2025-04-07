from aiogram import Router, types, F
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

from .commands_text import get_dict_with_command
from .preprocess import predict_category, all_categories
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup


class Questions(StatesGroup):
    question = State()
    answer = State()


router = Router()


@router.message(Command("start"))
async def start(message: types.Message):
    await message.answer(
        f"Привет, {message.from_user.first_name}!Этот бот сможет предложить тебе на выбор"
        f" те конторы, которые подойдут для твоего случая")


@router.message(Command("help"))
async def help_user(message: types.Message):
    ans = ""
    for key, value in get_dict_with_command().items():
        ans += key + ": " + value + "\n"
    await message.answer(ans)


@router.message(Command("categories"))
async def categories(message: types.Message):
    categor = all_categories()
    res = ''
    for key in categor:
        res += key +"\n"
    await message.answer(res)


@router.message(Command("question"))
async def question(message: types.Message, state: FSMContext):
    await message.answer("Введите ваш вопрос: ")
    await state.set_state(Questions.question)


@router.message(Questions.question)
async def answer(message: types.Message, state: FSMContext):
    ans = predict_category(message.text)
    await message.answer(f"Ваш ответ: {ans}. Желаете задать еще один вопрос?",
                         reply_markup=ReplyKeyboardMarkup(keyboard=[
                             [
                                 KeyboardButton(text="Yes"),
                                 KeyboardButton(text="No"),
                             ]
                         ]))
    await state.set_state(Questions.answer)


@router.message(Questions.answer, F.text.casefold() == "no")
async def answer_no(message: types.Message, state: FSMContext):
    await help_user(message)
    await state.set_state(None)


@router.message(Questions.answer, F.text.casefold() == "yes")
async def answer_yes(message: types.Message, state: FSMContext):
    await state.set_state(Questions.question)
    await question(message, state)


@router.message(Command("stop"))
async def stop(message: types.Message):
    exit(0)
