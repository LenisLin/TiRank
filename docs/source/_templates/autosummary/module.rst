{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :noindex:

{% if functions %}
.. rubric:: {{ _('Functions') }}

.. autosummary::
   :toctree:
   :nosignatures:

   {% for item in functions %}
      {%- if not item.startswith('_') %}
   {{ fullname }}.{{ item }}
      {%- endif %}
   {% endfor %}
{% endif %}

{% if classes %}
.. rubric:: {{ _('Classes') }}

.. autosummary::
   :toctree:
   :template: class.rst
   :nosignatures:

   {% for item in classes %}
      {%- if not item.startswith('_') %}
   {{ fullname }}.{{ item }}
      {%- endif %}
   {% endfor %}
{% endif %}