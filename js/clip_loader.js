import { app } from "../../scripts/app.js"

app.registerExtension({
	name: "ComfyUI_GGUF.clip_loader_gguf",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "CLIPLoaderGGUF") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;

			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);
				const node = this;

				const typeWidget = node.widgets.find((w) => w.name === "type");

				const updateWidgets = (type) => {
					const existingWidget = node.widgets.find((w) => w.name === "mmproj_path");
					if (existingWidget) {
						existingWidget.hidden = (type !== "qwen_image_edit")
					}
					const newSize = node.computeSize();
					node.size = newSize;
					app.graph.setDirtyCanvas(true, true);
				};
				typeWidget.callback = (value) => {
					updateWidgets(value);
				};
				setTimeout(() => {
					updateWidgets(typeWidget.value);
				}, 1);
			};
		}
	},
});
