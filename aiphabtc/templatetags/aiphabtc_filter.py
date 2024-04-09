from django import template

register = template.Library()

@register.filter
def sub(value, arg):
    return value - arg

@register.filter(name='get_item')
def get_item(list, index):
    try:
        return list[index]
    except IndexError:
        return None

@register.filter(name='to_float')
def to_float(value):
    try:
        return float(value)
    except ValueError:
        return 0.0


@register.filter(name='exclude_channels')
def exclude_channels(value, arg):
    """
    Custom template filter to exclude specific channels.

    Args:
        value (str): The channel to check.
        arg (str): A string of comma-separated channels to exclude.

    Returns:
        bool: True if the channel is not in the list to exclude, False otherwise.
    """
    exclude_list = arg.split(',')
    return value not in exclude_list
