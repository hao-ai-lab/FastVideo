import { app } from '../../../scripts/app.js'

function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existent object");
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property];
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r;
        };
    } else {
        object[property] = callback;
    }
}

function drawAutoAnnotated(ctx, node, widget_width, y, H) {
    const litegraph_base = LiteGraph;
    const show_text = app.canvas.ds.scale >= 0.5;
    const margin = 15;

    const autoTextWidth = 30;
    const autoTextRightMargin = 5;

    ctx.textAlign = 'left';
    ctx.strokeStyle = litegraph_base.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = litegraph_base.WIDGET_BGCOLOR;

    ctx.beginPath();
    if (show_text && ctx.roundRect) {
        ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.5]);
    } else {
        ctx.rect(margin, y, widget_width - margin * 2, H);
    }
    ctx.fill();

    if (show_text) {
        if (!this.disabled) ctx.stroke();
        const isAuto = this.isAuto === true;

        ctx.save();
        if (isAuto) {
            ctx.fillStyle = litegraph_base.WIDGET_TEXT_COLOR;
            ctx.strokeStyle = litegraph_base.WIDGET_TEXT_COLOR;
        } else {
            ctx.fillStyle = litegraph_base.WIDGET_SECONDARY_TEXT_COLOR;
            ctx.strokeStyle = litegraph_base.WIDGET_SECONDARY_TEXT_COLOR;
        }

        // Position for the cog
        const cogX = widget_width - autoTextRightMargin - autoTextWidth - 6;
        const cogY = y + H * 0.5;
        const cogRadius = 6;
        const toothLength = 2;
        const numTeeth = 8;
        const holeRadius = 2; // Radius of the center hole

        // Draw the cog
        ctx.beginPath();
        ctx.arc(cogX, cogY, cogRadius - toothLength, 0, Math.PI * 2);
        ctx.fill();

        // Draw the center hole (by clearing it)
        ctx.beginPath();
        ctx.arc(cogX, cogY, holeRadius, 0, Math.PI * 2);
        ctx.fillStyle = litegraph_base.WIDGET_BGCOLOR;
        ctx.fill();

        // Reset fill style for the teeth
        if (isAuto) {
            ctx.fillStyle = litegraph_base.WIDGET_TEXT_COLOR;
        } else {
            ctx.fillStyle = litegraph_base.WIDGET_SECONDARY_TEXT_COLOR;
        }

        // Draw teeth
        ctx.beginPath();
        for (let i = 0; i < numTeeth; i++) {
            const angle = (i / numTeeth) * Math.PI * 2;
            const innerX = cogX + (cogRadius - toothLength) * Math.cos(angle);
            const innerY = cogY + (cogRadius - toothLength) * Math.sin(angle);
            const outerX = cogX + cogRadius * Math.cos(angle);
            const outerY = cogY + cogRadius * Math.sin(angle);

            ctx.moveTo(innerX, innerY);
            ctx.lineTo(outerX, outerY);
        }
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.restore();

        // Draw label
        ctx.fillStyle = litegraph_base.WIDGET_SECONDARY_TEXT_COLOR;
        const label = this.label || this.name;
        if (label != null) {
            ctx.fillText(label, margin * 2 + 5, y + H * 0.7);
        }

        // Draw value
        ctx.textAlign = 'right';
        const text = isAuto ? "auto" : this.displayValue();
        ctx.fillStyle = isAuto ? litegraph_base.WIDGET_SECONDARY_TEXT_COLOR : litegraph_base.WIDGET_TEXT_COLOR;
        ctx.fillText(text, widget_width - autoTextRightMargin - autoTextWidth - 15, y + H * 0.7);

        // Draw increment/decrement buttons if not in AUTO mode and not a string widget
        if (!isAuto && !this.disabled && this.config[0] !== "FVAUTOSTRING") {
            // Draw decrement button (left triangle)
            ctx.fillStyle = litegraph_base.WIDGET_TEXT_COLOR;
            ctx.beginPath();
            ctx.moveTo(margin + 16, y + 5);
            ctx.lineTo(margin + 6, y + H * 0.5);
            ctx.lineTo(margin + 16, y + H - 5);
            ctx.fill();

            // Draw increment button (right triangle)
            ctx.beginPath();
            ctx.moveTo(widget_width - margin - 16, y + 5);
            ctx.lineTo(widget_width - margin - 6, y + H * 0.5);
            ctx.lineTo(widget_width - margin - 16, y + H - 5);
            ctx.fill();
        }
    }
}

function mouseAutoAnnotated(event, [x, y], node) {
    const widget_width = node.size[0];
    const margin = 15;
    const H = 20; // Widget height

    const autoTextWidth = 30;
    const autoTextRightMargin = 5;

    const cogRadius = 6;

    // Determine if clicking on increment/decrement buttons
    // For string widgets, don't handle increment/decrement
    const delta = this.config[0] === "FVAUTOSTRING" ? 0 :
        (x < 40 ? -1 : x > widget_width - 48 ? 1 : 0);
    const old_value = this.value;

    // Handle different event types
    if (event.type === "pointermove" && this.captureInput) {
        // Handle dragging
        const delta_move = (x - this.last_x) * 0.1 * (this.options.step || 1);
        this.last_x = x;

        if (this.isAuto) return false;

        // For combo boxes or string widgets, don't handle dragging
        if (this.config[0] === "FVAUTOCOMBO" || this.config[0] === "FVAUTOSTRING") return false;

        let v = parseFloat(this.value);
        v += delta_move;

        // Apply min/max constraints
        if (this.options.min != null) {
            v = Math.max(this.options.min, v);
        }
        if (this.options.max != null) {
            v = Math.min(this.options.max, v);
        }

        // Round to precision or to integer
        if (this.config[0] === "FVAUTOINT") {
            v = Math.round(v);
        } else if (this.options.precision !== undefined) {
            const precision = Math.pow(10, this.options.precision);
            v = Math.round(v * precision) / precision;
        }

        this.value = v;
        if (this.callback) {
            this.callback(this.value);
        }

        node.graph.setDirtyCanvas(true, false);
        return true;
    } else if (event.type === "pointerdown") {
        // Check if clicking on AUTO text - make region match the cog position and size
        if (x > widget_width - autoTextRightMargin - autoTextWidth - 5 - cogRadius &&
            x < widget_width - autoTextRightMargin - autoTextWidth - 5 + cogRadius) {
            this.isAuto = !this.isAuto;

            if (this.isAuto) {
                // Store current value before switching to auto
                this.cachedValue = this.value;
                this.value = -99999;  // Use the same special value
            } else {
                // Restore manual value when switching from auto
                this.value = this.cachedValue !== undefined ? this.cachedValue : (this.options.default || 0);
            }

            if (this.callback) {
                this.callback(this.value);
            }

            node.graph.setDirtyCanvas(true, false);
            return true;
        }

        if (this.isAuto) return false;

        // Handle increment/decrement buttons
        if (delta !== 0) {
            if (this.config[0] === "FVAUTOCOMBO") {
                // Get the combo options
                const options = this.options.values || [];
                if (options.length === 0) return true;

                // Find the current value in the options
                let currentIndex = -1;
                for (let i = 0; i < options.length; i++) {
                    const optValue = typeof options[i] === 'object' ? options[i].value : options[i];
                    if (optValue === this.value) {
                        currentIndex = i;
                        break;
                    }
                }

                // Calculate the new index
                let newIndex = currentIndex + delta;
                if (newIndex < 0) newIndex = options.length - 1;
                if (newIndex >= options.length) newIndex = 0;

                // Set the new value
                const newOption = options[newIndex];
                this.value = typeof newOption === 'object' ? newOption.value : newOption;

                if (this.callback) {
                    this.callback(this.value);
                }

                node.graph.setDirtyCanvas(true, false);
                return true;
            } else {
                // Original behavior for numeric widgets
                let v = parseFloat(this.value);
                v += delta * 0.1 * (this.options.step || 1);

                // Apply min/max constraints
                if (this.options.min != null) {
                    v = Math.max(this.options.min, v);
                }
                if (this.options.max != null) {
                    v = Math.min(this.options.max, v);
                }

                // Round to precision or to integer
                if (this.config[0] === "FVAUTOINT") {
                    v = Math.round(v);
                } else if (this.options.precision !== undefined) {
                    const precision = Math.pow(10, this.options.precision);
                    v = Math.round(v * precision) / precision;
                }

                this.value = v;
                if (this.callback) {
                    this.callback(this.value);
                }

                node.graph.setDirtyCanvas(true, false);
                return true;
            }
        }

        // Start dragging
        this.captureInput = true;
        this.last_x = x;
        return true;
    } else if (event.type === "pointerup") {
        // Handle click on value area
        if (event.click_time < 200 && delta === 0 && !this.isAuto) {
            if (this.config[0] === "FVAUTOCOMBO") {
                // Get the combo options from widget options
                const options = this.options.values || [];

                // Create menu items
                const menuItems = options.map(opt => {
                    // Handle both string options and {value, label} objects
                    const value = typeof opt === 'object' ? opt.value : opt;
                    const label = typeof opt === 'object' ? opt.label : opt.toString();

                    return {
                        content: label,
                        callback: () => {
                            this.value = value;

                            if (this.callback) {
                                this.callback(this.value);
                            }

                            node.graph.setDirtyCanvas(true, false);
                        }
                    };
                });

                // Show the context menu
                new LiteGraph.ContextMenu(menuItems, {
                    event: event,
                    title: null,
                    callback: null,
                    extra: node
                });

                return true;
            } else {
                // Original prompt for other types
                const d_callback = (v) => {
                    if (this.config[0] === "FVAUTOSTRING") {
                        // For string widgets, just use the value as is
                        this.value = v;
                    } else {
                        this.value = this.parseValue?.(v) ?? Number(v);

                        // Apply min/max constraints
                        if (this.options.min != null) {
                            this.value = Math.max(this.options.min, this.value);
                        }
                        if (this.options.max != null) {
                            this.value = Math.min(this.options.max, this.value);
                        }

                        // Round to precision or to integer
                        if (this.config[0] === "FVAUTOINT") {
                            this.value = Math.round(this.value);
                        } else if (this.options.precision !== undefined) {
                            const precision = Math.pow(10, this.options.precision);
                            this.value = Math.round(this.value * precision) / precision;
                        }
                    }

                    if (this.callback) {
                        this.callback(this.value);
                    }

                    node.graph.setDirtyCanvas(true, false);
                };

                const dialog = app.canvas.prompt(
                    'Value',
                    this.value,
                    d_callback,
                    event
                );
            }
            return true;
        }

        this.captureInput = false;
        return true;
    }

    return false;
}

function makeAutoAnnotated(widget, inputData) {
    const original = {
        callback: widget.callback,
        type: widget.type,
        value: widget.value
    };

    // Add AUTO properties to the widget
    Object.assign(widget, {
        type: "BOOLEAN",
        draw: drawAutoAnnotated,
        mouse: mouseAutoAnnotated,
        isAuto: true,
        cachedValue: widget.value,
        captureInput: false,
        last_x: 0,
        computeSize(width) {
            return [width, 20];
        },
        displayValue: function () {
            if (this.config[0] === "FVAUTOINT") {
                return Math.round(this.value).toString();
            }
            if (this.config[0] === "FVAUTOCOMBO") {
                return this.value;
            }
            if (this.config[0] === "FVAUTOSTRING") {
                return this.value;
            }
            // For FLOAT values, check if it's actually an integer
            if (Number.isInteger(this.value)) {
                return this.value.toString();
            }
            return this.value.toFixed(this.options.precision || 2);
        },
        parseValue: function (v) {
            if (this.config[0] === "FVAUTOSTRING") {
                return v;
            }
            if (typeof v === "string") {
                return parseFloat(v);
            }
            return v;
        },
        serializeValue: function () {
            // Return special value for AUTO mode
            return this.isAuto ? -99999 : this.value;
        },
        deserializeValue: function (data) {
            if (data === -99999) {
                this.isAuto = true;
                this.value = -99999;
            } else {
                this.isAuto = false;
                this.value = data;
                this.cachedValue = data;
            }
        },
        config: inputData,
        options: Object.assign({}, inputData[1], widget.options),
        original: original  // Store original properties for reference
    });

    // Override callback to handle AUTO mode
    widget.callback = function (v) {
        if (this.isAuto) {
            return; // Don't call the original callback in AUTO mode
        }
        return original.callback?.call(this, v);
    };

    return widget;
}

app.registerExtension({
    name: "FastVideo.AutoWidgets",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name == "VideoGenerator" || nodeData?.name === "InferenceArgs" || nodeData?.name === "VAEConfig" ||
            nodeData?.name === "TextEncoderConfig" || nodeData?.name === "DITConfig") {
            // Add serialization support
            chainCallback(nodeType.prototype, "onSerialize", function (info) {
                if (!this.widgets) {
                    return;
                }
                // Ensure widgets_values exists
                if (!info.widgets_values) {
                    info.widgets_values = {};
                }

                // Store AUTO widget states in a separate property
                if (!info.auto_widget_states) {
                    info.auto_widget_states = {};
                }

                // Handle AUTO widgets specially
                for (const w of this.widgets) {
                    if (w.type === "BOOLEAN" && w.isAuto !== undefined) {
                        // Store the serialized value (for Python node)
                        info.widgets_values[w.name] = w.serializeValue();

                        // Store the full state (for UI restoration)
                        info.auto_widget_states[w.name] = {
                            isAuto: w.isAuto,
                            value: w.value,
                            cachedValue: w.cachedValue
                        };
                    }
                }
            });

            // Add deserialization support
            chainCallback(nodeType.prototype, "onConfigure", function (info) {
                if (!this.widgets) {
                    return;
                }

                // First, restore from widgets_values (for backward compatibility)
                if (info.widgets_values && Array.isArray(info.widgets_values)) {
                    for (let i = 0; i < this.widgets.length && i < info.widgets_values.length; i++) {
                        const w = this.widgets[i];
                        const value = info.widgets_values[i];

                        if (w.type === "BOOLEAN" && w.isAuto !== undefined) {
                            w.deserializeValue(value);
                        }
                    }
                }

                // Then, restore full state if available
                if (info.auto_widget_states) {
                    for (const w of this.widgets) {
                        if (w.type === "BOOLEAN" && w.isAuto !== undefined && w.name in info.auto_widget_states) {
                            const state = info.auto_widget_states[w.name];

                            w.isAuto = state.isAuto;
                            w.cachedValue = state.cachedValue;
                            w.value = state.isAuto ? -99999 : state.value;

                            w.callback?.(w.value);
                        }
                    }
                }

                // Force a redraw
                this.graph?.setDirtyCanvas(true, true);
            });

            // Override onDrawForeground to ensure our widgets are drawn
            chainCallback(nodeType.prototype, "onDrawForeground", function (ctx) {
                // Check if we have any AUTO widgets
                if (this.widgets) {
                    const autoWidgets = this.widgets.filter(w => w.type === "BOOLEAN" && w.isAuto !== undefined);
                    // Debug: Check if draw methods are properly set
                    for (const w of autoWidgets) {
                        console.log(`Widget ${w.name} draw method:`, w.draw === drawAutoAnnotated ? "Correctly set" : "NOT SET CORRECTLY");
                    }
                }
            });

            // Override addInput to handle AUTO widgets
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                // Convert any existing widgets to AUTO widgets if needed
                let new_widgets = [];
                const intWidgetNames = ["sp_size", "tp_size", "height", "width", "num_frames", "num_inference_steps", "flow_shift", "seed", "fps", "scale_factor",
                    "tile_sample_min_height", "tile_sample_min_width", "tile_sample_min_num_frames", "tile_sample_stride_height", "tile_sample_stride_width",
                    "tile_sample_stride_num_frames", "blend_num_frames"
                ]
                const floatWidgetNames = ["embedded_cfg_scale", "guidance_scale"]
                const comboWidgetNames = ["vae_tiling", "vae_precision", "vae_sp", "text_encoder_precision", "precision",
                    "load_encoder", "load_decoder", "use_tiling", "use_temporal_tiling", "use_parallel_tiling", "use_cpu_offload", "enable_teacache"
                ]
                const stringWidgetNames = ["prefix", "quant_config", "lora_config", "image_path"]

                if (this.widgets) {
                    for (let w of this.widgets) {
                        if (intWidgetNames.includes(w.name)) {
                            new_widgets.push(makeAutoAnnotated(w, ["FVAUTOINT", { "default": 0 }]));
                        } else if (floatWidgetNames.includes(w.name)) {
                            new_widgets.push(makeAutoAnnotated(w, ["FVAUTOFLOAT", { "default": 0 }]));
                        } else if (comboWidgetNames.includes(w.name)) {
                            new_widgets.push(makeAutoAnnotated(w, ["FVAUTOCOMBO", { "default": 0 }]));
                        } else if (stringWidgetNames.includes(w.name)) {
                            new_widgets.push(makeAutoAnnotated(w, ["FVAUTOSTRING", { "default": "" }]));
                        } else {
                            new_widgets.push(w);
                        }
                    }
                    this.widgets = new_widgets;

                    const autoWidgets = this.widgets.filter(w => w.type === "BOOLEAN" && w.isAuto !== undefined);
                }

                this.graph?.setDirtyCanvas(true, true);
            });
        }
    },

    async init() {
        // Force a redraw of all nodes when the extension initializes
        if (app.graph) {
            setTimeout(() => {
                app.graph.setDirtyCanvas(true, true);
            }, 1000);
        }
    }
});

console.log("FastVideo.core.js loaded");
