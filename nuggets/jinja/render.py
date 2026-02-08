import json
import jinja2

with open('c.json') as f:
    data = json.load(f)

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader('.')
)
template = env.get_template('b.html.j2')
with open('d.html', "w") as f:
    f.write(template.render(**data))
