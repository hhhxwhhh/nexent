from langchain_core.tools import tool, StructuredTool
from typing import Annotated, Union
from pydantic import BaseModel, Field
import math
import logging

logger = logging.getLogger(__name__)

@tool
def add(a: int, b: int) -> int:
    """
    Calculate the sum of two numbers.
    
    Args:
        a (int): The first number.
        b (int): The second number.
        
    Returns:
        int: The sum of a and b.
    """
    result = a + b
    logger.info(f"Addition: {a} + {b} = {result}")
    return result

@tool
def multiply(
        a: Annotated[int, "scale factor"],
        b: Annotated[int, "scale factor"],
) -> int:
    """
    Calculate the product of two integers.
    
    Args:
        a (int): The first number.
        b (int): The second number.
        
    Returns:
        int: The product of a and b.
    """
    result = a * b
    logger.info(f"Multiplication: {a} * {b} = {result}")
    return result

@tool(parse_docstring=True)
def subtraction(a: int, b: int) -> int:
    """
    Calculate the difference between two numbers.

    Args:
    a (int): The first number.
    b (int): The second number.

    Returns:
    int: The difference between a and b.
    """
    result = a - b
    logger.info(f"Subtraction: {a} - {b} = {result}")
    return result


class DivisionInput(BaseModel):
    num1: int = Field(description="first number")
    num2: int = Field(description="second number")

@tool("division", args_schema=DivisionInput)
def division(num1: int, num2: int) -> Union[int, float]:
    """
    Return the quotient of two numbers.
    
    Args:
        num1 (int): The dividend.
        num2 (int): The divisor.
        
    Returns:
        Union[int, float]: The quotient of num1 and num2.
        
    Raises:
        ZeroDivisionError: If num2 is zero.
    """
    if num2 == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    
    result = num1 / num2
    logger.info(f"Division: {num1} / {num2} = {result}")
    return result


class ExponentiationInput(BaseModel):
    num: int = Field(description="base number")
    power: int = Field(description="exponent")

def exponentiation_func(num: int, power: int) -> int:
    """
    Calculate the power of a number
    
    Args:
        num (int): The base number.
        power (int): The exponent.
        
    Returns:
        int: The result of num raised to the power.
    """
    result = num ** power
    logger.info(f"Exponentiation: {num} ^ {power} = {result}")
    return result

exponentiation = StructuredTool.from_function(
    func=exponentiation_func,
    name="exponentiation",
    description="Calculate the power of a number",
    args_schema=ExponentiationInput,
    return_direct=True,
)

class SquareRootInput(BaseModel):
    number: float = Field(description="number to calculate square root of")

@tool("square_root", args_schema=SquareRootInput)
def square_root(number: float) -> float:
    """
    Calculate the square root of a number.
    
    Args:
        number (float): The number to calculate square root of.
        
    Returns:
        float: The square root of the number.
        
    Raises:
        ValueError: If number is negative.
    """
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    
    result = math.sqrt(number)
    logger.info(f"Square root: sqrt({number}) = {result}")
    return result

class PercentageInput(BaseModel):
    part: float = Field(description="part value")
    whole: float = Field(description="whole value")

@tool("percentage", args_schema=PercentageInput)
def percentage(part: float, whole: float) -> float:
    """
    Calculate the percentage of part relative to whole.
    
    Args:
        part (float): The part value.
        whole (float): The whole value.
        
    Returns:
        float: The percentage (0-100).
        
    Raises:
        ZeroDivisionError: If whole is zero.
    """
    if whole == 0:
        raise ZeroDivisionError("Cannot calculate percentage with zero as whole")
    
    result = (part / whole) * 100
    logger.info(f"Percentage: ({part} / {whole}) * 100 = {result}%")
    return result

# List of all tools
COMPUTE_TOOLS = [
    add,
    multiply,
    subtraction,
    division,
    exponentiation,
    square_root,
    percentage
]

if __name__ == "__main__":
    # Example usage
    print("Compute tools available:")
    for tool in COMPUTE_TOOLS:
        print(f"- {tool.name}: {tool.description}")