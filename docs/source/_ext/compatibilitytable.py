from docutils import nodes
from docutils.parsers.rst import Directive

import yaml

class CompatibilityTable(Directive):
    required_arguments = 1

    def run(self):
        # "documentation":
        # https://github.com/docutils/docutils/blob/173189b4c1c095a43c9388f4edd9bf1ff5d5b49d/docutils/docutils/parsers/rst/states.py#L1793
        # https://github.com/docutils/docutils/blob/173189b4c1c095a43c9388f4edd9bf1ff5d5b49d/docutils/docutils/nodes.py#L439
        
        releases = None
        with open(self.arguments[0]) as f:
            releases = yaml.safe_load(f)
        
        table = nodes.table()
        tgroup = nodes.tgroup(cols=2)
        for _ in range(len(releases["components"])+1):
            colspec = nodes.colspec(colwidth=1)
            tgroup.append(colspec)
        table += tgroup

        thead = nodes.thead()
        tgroup += thead
        row = nodes.row()
        for component in ["release"] + releases["components"]:
            entry = nodes.entry()
            entry += nodes.paragraph(text=component)
            row += entry

        thead.append(row)
        
        tbody = nodes.tbody()
        for release in releases["releases"]:
            row = nodes.row()
            row += nodes.entry()
            row[0] += nodes.strong(text=release["name"])
            for component_name in releases["components"]:
                component = release["components"][component_name]
                entry = nodes.entry()
                para = nodes.paragraph()
                para += nodes.reference(text=component["version"], refuri=component["link"], internal=False)
                if "helm" in component:
                    para += nodes.Text(" (helm ")
                    para += nodes.reference(text=component["helm"]["version"], refuri=component["link"], internal=False)
                    para += nodes.Text(")")
                entry += para
                row += entry
            tbody += row
        tgroup += tbody

        return [table]

def setup(app):
    app.add_directive("compatibilitytable", CompatibilityTable)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }