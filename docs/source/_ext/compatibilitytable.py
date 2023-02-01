import yaml
from docutils import nodes
from docutils.parsers.rst import Directive


def has_helm_chart(table: dict, component: str) -> bool:
    for release in table["releases"]:
        if component in release["components"]:
            if "helm" in release["components"][component]:
                return True
    return False


class CompatibilityTable(Directive):
    required_arguments = 1

    def run(self):
        # "documentation":
        # https://docutils.sourceforge.io/docs/ref/doctree.html#table
        # https://github.com/docutils/docutils/blob/173189b4c1c095a43c9388f4edd9bf1ff5d5b49d/docutils/docutils/parsers/rst/states.py#L1793
        # https://github.com/docutils/docutils/blob/173189b4c1c095a43c9388f4edd9bf1ff5d5b49d/docutils/docutils/nodes.py#L439

        # documentation says nodes can be constructed by passing their children to the constructor
        # for instance nodes.entry(nodes.Text("lol")) should work
        # but it doesn't
        # this leads to needing to first create the node and then attach children to it

        releases = None
        with open(self.arguments[0]) as f:
            releases = yaml.safe_load(f)

        table = nodes.table()
        tgroup = nodes.tgroup()
        for _ in range((len(releases["components"]) + 1) * 2):
            colspec = nodes.colspec(colwidth=1)
            tgroup.append(colspec)
        table += tgroup

        thead = nodes.thead()
        tgroup += thead
        component_row = nodes.row()
        helm_row = nodes.row()

        for component in ["release"] + releases["components"]:
            if not has_helm_chart(releases, component):
                name_entry = nodes.entry(morerows=1, morecols=1)
            else:
                name_entry = nodes.entry(morecols=1)
                helm_row += [nodes.entry(), nodes.entry()]
                helm_row[-2] += nodes.paragraph(text="app")
                helm_row[-1] += nodes.emphasis(text="helm")

            name_entry += nodes.paragraph(text=component)
            component_row += name_entry

        thead.append(component_row)
        thead.append(helm_row)

        tbody = nodes.tbody()
        for release in releases["releases"]:
            row = nodes.row()
            row += nodes.entry(morecols=1)
            row[0] += nodes.strong(text=release["name"])
            for component_name in releases["components"]:
                component = release["components"][component_name]
                app_para = nodes.paragraph()
                app_para += nodes.reference(
                    text=component["version"], refuri=component["link"], internal=False
                )
                if "helm" in component:
                    row += nodes.entry()
                    row[-1] += app_para
                    para = nodes.emphasis()
                    para += nodes.reference(
                        text=component["helm"]["version"],
                        refuri=component["helm"]["link"],
                        internal=False,
                    )
                    row += nodes.entry()
                    row[-1] += para
                else:
                    row += nodes.entry(morecols=1)
                    row[-1] += app_para

            tbody += row
        tgroup += tbody

        return [table]


def setup(app):
    app.add_directive("compatibilitytable", CompatibilityTable)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
