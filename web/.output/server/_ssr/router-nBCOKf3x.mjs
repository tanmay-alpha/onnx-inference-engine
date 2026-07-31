import { i as __toESM } from "../_runtime.mjs";
import { n as useForm, r as require_react, t as u } from "../_libs/@hookform/resolvers+[...].mjs";
import { A as redirect, _ as useNavigate, c as HeadContent, d as createRouter, f as Outlet, g as Link, h as createRootRouteWithContext, m as createFileRoute, p as lazyRouteComponent, s as Scripts, v as useRouter } from "../_libs/@tanstack/react-router+[...].mjs";
import { n as require_jsx_runtime } from "../_libs/radix-ui__react-context+react.mjs";
import { a as DialogOverlay$1, i as DialogDescription$1, n as DialogClose, o as DialogPortal$1, r as DialogContent$1, s as DialogTitle$1, t as Dialog$1 } from "../_libs/@radix-ui/react-dialog+[...].mjs";
import { A as Cpu, D as FileUp, I as Check, M as Clock, N as CircleAlert, O as Eye, R as ArrowRight, S as Inbox, a as TriangleAlert, b as Key, c as Shield, d as RefreshCw, f as Plus, g as LogIn, i as Upload, j as Copy, k as EyeOff, n as X, p as Play, r as UserPlus, s as Trash2, v as LoaderCircle, w as HardDrive, x as Info } from "../_libs/lucide-react.mjs";
import { d as revokeApiKey, i as fetchModels, l as login, n as createApiKey, p as uploadModel, r as deleteModel, s as listApiKeys, t as CrucibleLayout, u as register } from "./Layout-CgtWwv96.mjs";
import { t as cva } from "../_libs/class-variance-authority+clsx.mjs";
import { n as cn, t as Button } from "./button-DRsC1qZi.mjs";
import { t as QueryClient } from "../_libs/tanstack__query-core.mjs";
import { t as QueryClientProvider } from "../_libs/tanstack__react-query.mjs";
import { n as stringType, t as objectType } from "../_libs/zod.mjs";
import { t as Root } from "../_libs/radix-ui__react-label.mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/router-nBCOKf3x.js
var import_react = /* @__PURE__ */ __toESM(require_react());
var import_jsx_runtime = require_jsx_runtime();
var styles_default = "/assets/styles-BY_3sGxz.css";
var crucible_default = "/assets/crucible-DzLgedlU.css";
function reportLovableError(error, context = {}) {
	if (typeof window === "undefined") return;
	window.__lovableEvents?.captureException?.(error, {
		source: "react_error_boundary",
		route: window.location.pathname,
		...context
	}, {
		mechanism: "react_error_boundary",
		handled: false,
		severity: "error"
	});
}
function NotFoundComponent() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
		className: "flex min-h-screen items-center justify-center bg-background px-4",
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "max-w-md text-center",
			children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
					className: "text-7xl font-bold text-foreground",
					children: "404"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
					className: "mt-4 text-xl font-semibold text-foreground",
					children: "Page not found"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-2 text-sm text-muted-foreground",
					children: "The page you're looking for doesn't exist or has been moved."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					className: "mt-6",
					children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
						to: "/",
						className: "inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90",
						children: "Go home"
					})
				})
			]
		})
	});
}
function ErrorComponent({ error, reset }) {
	console.error(error);
	const router = useRouter();
	(0, import_react.useEffect)(() => {
		reportLovableError(error, { boundary: "tanstack_root_error_component" });
	}, [error]);
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
		className: "flex min-h-screen items-center justify-center bg-background px-4",
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "max-w-md text-center",
			children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
					className: "text-xl font-semibold tracking-tight text-foreground",
					children: "This page didn't load"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-2 text-sm text-muted-foreground",
					children: "Something went wrong on our end. You can try refreshing or head back home."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "mt-6 flex flex-wrap justify-center gap-2",
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
						onClick: () => {
							router.invalidate();
							reset();
						},
						className: "inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90",
						children: "Try again"
					}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
						href: "/",
						className: "inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent",
						children: "Go home"
					})]
				})
			]
		})
	});
}
var Route$13 = createRootRouteWithContext()({
	head: () => ({
		meta: [
			{ charSet: "utf-8" },
			{
				name: "viewport",
				content: "width=device-width, initial-scale=1"
			},
			{ title: "Crucible — ONNX inference engine for the browser" },
			{
				name: "description",
				content: "A from-scratch ONNX runtime. C++17 core with a hand-written protobuf decoder and Kahn's-algorithm graph executor. Pure-Rust reimplementation compiled to WebAssembly. Fraud inference in the browser tab."
			},
			{
				name: "author",
				content: "Crucible"
			},
			{
				property: "og:title",
				content: "Crucible — ONNX inference engine for the browser"
			},
			{
				property: "og:description",
				content: "3.1 MB WASM runtime. Zero server calls. C++ core, Rust WASM build, honest benchmarks."
			},
			{
				property: "og:type",
				content: "website"
			},
			{
				property: "og:site_name",
				content: "Crucible"
			},
			{
				name: "twitter:card",
				content: "summary_large_image"
			},
			{
				name: "theme-color",
				content: "#FAFAF7"
			}
		],
		links: [
			{
				rel: "preconnect",
				href: "https://fonts.googleapis.com"
			},
			{
				rel: "preconnect",
				href: "https://fonts.gstatic.com",
				crossOrigin: "anonymous"
			},
			{
				rel: "stylesheet",
				href: "https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Newsreader:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap"
			},
			{
				rel: "stylesheet",
				href: styles_default
			},
			{
				rel: "stylesheet",
				href: crucible_default
			},
			{
				rel: "icon",
				href: "/favicon.svg",
				type: "image/svg+xml"
			}
		]
	}),
	shellComponent: RootShell,
	component: RootComponent,
	notFoundComponent: NotFoundComponent,
	errorComponent: ErrorComponent
});
function RootShell({ children }) {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("html", {
		lang: "en",
		children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("head", { children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(HeadContent, {}) }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("body", { children: [children, /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Scripts, {})] })]
	});
}
function RootComponent() {
	const { queryClient } = Route$13.useRouteContext();
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(QueryClientProvider, {
		client: queryClient,
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Outlet, {})
	});
}
var $$splitComponentImporter$8 = () => import("./story-7bxRuW56.mjs");
var Route$12 = createFileRoute("/story")({
	head: () => ({
		meta: [
			{ title: "Story · Crucible" },
			{
				name: "description",
				content: "Why Crucible exists: ONNX Runtime is 50MB+, PyTorch is 750MB+, and neither fits in a browser tab. So I wrote it from scratch."
			},
			{
				property: "og:title",
				content: "Crucible — Why This Exists"
			},
			{
				property: "og:description",
				content: "A first-person build story: hand-writing an ONNX protobuf decoder and keeping a C++ engine and a Rust engine bit-for-bit identical."
			},
			{
				property: "og:url",
				content: "/story"
			}
		],
		links: [{
			rel: "canonical",
			href: "/story"
		}]
	}),
	component: lazyRouteComponent($$splitComponentImporter$8, "component")
});
var $$splitComponentImporter$7 = () => import("./roadmap-CGzpto4G.mjs");
var Route$11 = createFileRoute("/roadmap")({
	head: () => ({
		meta: [
			{ title: "Roadmap · Crucible" },
			{
				name: "description",
				content: "What's shipped, in progress, and planned for Crucible — from the pure-Rust WASM build to biometric matching and OCR."
			},
			{
				property: "og:title",
				content: "Crucible — Roadmap"
			},
			{
				property: "og:description",
				content: "Active development: kernel parity, int8 quantization, biometrics, OCR, WebGPU."
			},
			{
				property: "og:url",
				content: "/roadmap"
			}
		],
		links: [{
			rel: "canonical",
			href: "/roadmap"
		}]
	}),
	component: lazyRouteComponent($$splitComponentImporter$7, "component")
});
var Input = import_react.forwardRef(({ className, type, ...props }, ref) => {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)("input", {
		type,
		className: cn("flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-base shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 md:text-sm", className),
		ref,
		...props
	});
});
Input.displayName = "Input";
var labelVariants = cva("text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70");
var Label = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Root, {
	ref,
	className: cn(labelVariants(), className),
	...props
}));
Label.displayName = Root.displayName;
var Card = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
	ref,
	className: cn("rounded-xl border bg-card text-card-foreground shadow", className),
	...props
}));
Card.displayName = "Card";
var CardHeader = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
	ref,
	className: cn("flex flex-col space-y-1.5 p-6", className),
	...props
}));
CardHeader.displayName = "CardHeader";
var CardTitle = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
	ref,
	className: cn("font-semibold leading-none tracking-tight", className),
	...props
}));
CardTitle.displayName = "CardTitle";
var CardDescription = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
	ref,
	className: cn("text-sm text-muted-foreground", className),
	...props
}));
CardDescription.displayName = "CardDescription";
var CardContent = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
	ref,
	className: cn("p-6 pt-0", className),
	...props
}));
CardContent.displayName = "CardContent";
var CardFooter = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
	ref,
	className: cn("flex items-center p-6 pt-0", className),
	...props
}));
CardFooter.displayName = "CardFooter";
var registerSchema = objectType({
	fullName: stringType().min(1, "Full name is required"),
	email: stringType().min(1, "Email is required").email("Please enter a valid email"),
	password: stringType().min(8, "Password must be at least 8 characters"),
	confirmPassword: stringType().min(1, "Please confirm your password")
}).refine((data) => data.password === data.confirmPassword, {
	message: "Passwords do not match",
	path: ["confirmPassword"]
});
function RegisterPage() {
	const navigate = useNavigate();
	const [showPassword, setShowPassword] = (0, import_react.useState)(false);
	const [showConfirm, setShowConfirm] = (0, import_react.useState)(false);
	const [error, setError] = (0, import_react.useState)(null);
	const [loading, setLoading] = (0, import_react.useState)(false);
	const [success, setSuccess] = (0, import_react.useState)(false);
	const { register: register$1, handleSubmit, formState: { errors, isSubmitting } } = useForm({ resolver: u(registerSchema) });
	const onSubmit = async (data) => {
		setLoading(true);
		setError(null);
		try {
			await register(data.email, data.password, data.fullName);
			setSuccess(true);
			setTimeout(() => {
				navigate({ to: "/dashboard" });
			}, 1200);
		} catch (err) {
			setError(err instanceof Error ? err.message : "Registration failed. Please try again.");
		} finally {
			setLoading(false);
		}
	};
	if (success) return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CrucibleLayout, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
		className: "c-auth-page",
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Card, {
			className: "c-auth-card",
			children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CardContent, {
				style: {
					textAlign: "center",
					padding: "40px 24px"
				},
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						style: {
							width: 48,
							height: 48,
							borderRadius: "50%",
							background: "#ddeedf",
							border: "1px solid #b4d8be",
							color: "#166534",
							display: "inline-flex",
							alignItems: "center",
							justifyContent: "center",
							marginBottom: 16
						},
						children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Check, { size: 24 })
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
						style: {
							fontFamily: "var(--f-serif)",
							fontSize: 24,
							fontWeight: 500,
							color: "var(--ink)",
							marginBottom: 8
						},
						children: "Account created"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						style: {
							color: "var(--ink-muted)",
							fontSize: 14
						},
						children: "Redirecting to your dashboard..."
					})
				]
			})
		})
	}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("style", { children: `
          .c-auth-page {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 60vh;
            padding: 48px 20px;
          }
          .c-auth-card {
            width: 100%;
            max-width: 420px;
          }
        ` })] });
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CrucibleLayout, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
		className: "c-auth-page",
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Card, {
			className: "c-auth-card",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CardHeader, {
				className: "c-auth-header",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						className: "c-auth-icon",
						children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(UserPlus, { size: 22 })
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CardTitle, {
						className: "c-auth-title",
						children: "Create account"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CardDescription, {
						className: "c-auth-desc",
						children: "Start running ONNX inference and fraud detection"
					})
				]
			}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CardContent, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("form", {
				onSubmit: handleSubmit(onSubmit),
				className: "c-auth-form",
				children: [
					error && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-auth-error",
						role: "alert",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CircleAlert, { size: 15 }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { children: error })]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-auth-field",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Label, {
								htmlFor: "fullName",
								children: "Full name"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Input, {
								id: "fullName",
								type: "text",
								placeholder: "Your name",
								autoComplete: "name",
								...register$1("fullName"),
								className: errors.fullName ? "c-input-error" : ""
							}),
							errors.fullName && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "c-field-error",
								children: errors.fullName.message
							})
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-auth-field",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Label, {
								htmlFor: "email",
								children: "Email"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Input, {
								id: "email",
								type: "email",
								placeholder: "you@example.com",
								autoComplete: "email",
								...register$1("email"),
								className: errors.email ? "c-input-error" : ""
							}),
							errors.email && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "c-field-error",
								children: errors.email.message
							})
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-auth-field",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Label, {
								htmlFor: "password",
								children: "Password"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-password-wrap",
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Input, {
									id: "password",
									type: showPassword ? "text" : "password",
									placeholder: "At least 8 characters",
									autoComplete: "new-password",
									...register$1("password"),
									className: errors.password ? "c-input-error" : ""
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
									type: "button",
									className: "c-password-toggle",
									onClick: () => setShowPassword(!showPassword),
									"aria-label": showPassword ? "Hide password" : "Show password",
									children: showPassword ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)(EyeOff, { size: 15 }) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Eye, { size: 15 })
								})]
							}),
							errors.password && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "c-field-error",
								children: errors.password.message
							})
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-auth-field",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Label, {
								htmlFor: "confirmPassword",
								children: "Confirm password"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-password-wrap",
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Input, {
									id: "confirmPassword",
									type: showConfirm ? "text" : "password",
									placeholder: "Repeat your password",
									autoComplete: "new-password",
									...register$1("confirmPassword"),
									className: errors.confirmPassword ? "c-input-error" : ""
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
									type: "button",
									className: "c-password-toggle",
									onClick: () => setShowConfirm(!showConfirm),
									"aria-label": showConfirm ? "Hide password" : "Show password",
									children: showConfirm ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)(EyeOff, { size: 15 }) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Eye, { size: 15 })
								})]
							}),
							errors.confirmPassword && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "c-field-error",
								children: errors.confirmPassword.message
							})
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Button, {
						type: "submit",
						className: "c-auth-submit",
						disabled: isSubmitting || loading,
						children: loading || isSubmitting ? /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(import_jsx_runtime.Fragment, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(LoaderCircle, {
							size: 16,
							className: "c-spin"
						}), "Creating account..."] }) : /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(import_jsx_runtime.Fragment, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(UserPlus, { size: 16 }), "Create account"] })
					})
				]
			}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
				className: "c-auth-footer",
				children: [
					"Already have an account?",
					" ",
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
						to: "/login",
						className: "c-auth-link",
						children: "Sign in"
					})
				]
			})] })]
		})
	}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("style", { children: `
        .c-auth-page {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 60vh;
          padding: 48px 20px;
        }
        .c-auth-card {
          width: 100%;
          max-width: 420px;
        }
        .c-auth-header {
          text-align: center;
          padding-bottom: 20px;
        }
        .c-auth-icon {
          width: 44px;
          height: 44px;
          border-radius: 50%;
          background: #fce8d5;
          border: 1px solid #ebc6a3;
          color: #c2410c;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 14px;
        }
        .c-auth-title {
          font-family: var(--f-serif);
          font-size: 28px;
          font-weight: 500;
          color: var(--ink);
          letter-spacing: -0.02em;
        }
        .c-auth-desc {
          color: var(--ink-muted);
          font-size: 14px;
          margin-top: 6px;
        }
        .c-auth-form {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .c-auth-field {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }
        .c-password-wrap {
          position: relative;
        }
        .c-password-wrap .c-input {
          padding-right: 40px;
        }
        .c-password-toggle {
          position: absolute;
          right: 10px;
          top: 50%;
          transform: translateY(-50%);
          background: none;
          border: none;
          cursor: pointer;
          color: var(--ink-muted);
          padding: 4px;
          display: flex;
          align-items: center;
        }
        .c-password-toggle:hover {
          color: var(--ink);
        }
        .c-input-error {
          border-color: var(--risk) !important;
        }
        .c-field-error {
          font-size: 12px;
          color: var(--risk);
          font-family: var(--f-mono);
        }
        .c-auth-error {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 12px;
          border-radius: 2px;
          background: #fadcdc;
          border: 1px solid #e4b4b4;
          color: var(--risk);
          font-size: 13px;
          line-height: 1.4;
        }
        .c-auth-submit {
          width: 100%;
          justify-content: center;
          margin-top: 4px;
        }
        .c-auth-footer {
          text-align: center;
          margin-top: 18px;
          font-size: 13px;
          color: var(--ink-muted);
        }
        .c-auth-link {
          color: var(--trace);
          font-weight: 600;
          text-decoration: underline;
          text-underline-offset: 2px;
        }
        .c-auth-link:hover {
          color: var(--forge-deep);
        }
        .c-spin {
          animation: c-spin 1s linear infinite;
        }
        @keyframes c-spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      ` })] });
}
var Route$10 = { component: RegisterPage };
var $$splitComponentImporter$6 = () => import("./playground-Cx4NgPHu.mjs");
var Route$9 = createFileRoute("/playground")({
	head: () => ({ meta: [
		{ title: "Playground · Crucible" },
		{
			name: "description",
			content: "Drop an ONNX model, feed a tensor, inspect the graph — all in the browser."
		},
		{
			property: "og:title",
			content: "Crucible WASM Playground"
		},
		{
			property: "og:description",
			content: "Interactive ONNX runtime running entirely client-side."
		}
	] }),
	component: lazyRouteComponent($$splitComponentImporter$6, "component")
});
var Table = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
	className: "relative w-full overflow-auto",
	children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)("table", {
		ref,
		className: cn("w-full caption-bottom text-sm", className),
		...props
	})
}));
Table.displayName = "Table";
var TableHeader = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("thead", {
	ref,
	className: cn("[&_tr]:border-b", className),
	...props
}));
TableHeader.displayName = "TableHeader";
var TableBody = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("tbody", {
	ref,
	className: cn("[&_tr:last-child]:border-0", className),
	...props
}));
TableBody.displayName = "TableBody";
var TableFooter = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("tfoot", {
	ref,
	className: cn("border-t bg-muted/50 font-medium [&>tr]:last:border-b-0", className),
	...props
}));
TableFooter.displayName = "TableFooter";
var TableRow = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("tr", {
	ref,
	className: cn("border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted", className),
	...props
}));
TableRow.displayName = "TableRow";
var TableHead = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", {
	ref,
	className: cn("h-10 px-2 text-left align-middle font-medium text-muted-foreground [&:has([role=checkbox])]:pr-0 [&>[role=checkbox]]:translate-y-[2px]", className),
	...props
}));
TableHead.displayName = "TableHead";
var TableCell = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
	ref,
	className: cn("p-2 align-middle [&:has([role=checkbox])]:pr-0 [&>[role=checkbox]]:translate-y-[2px]", className),
	...props
}));
TableCell.displayName = "TableCell";
var TableCaption = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("caption", {
	ref,
	className: cn("mt-4 text-sm text-muted-foreground", className),
	...props
}));
TableCaption.displayName = "TableCaption";
var fmtBytes = (b) => {
	if (b < 1024) return `${b} B`;
	if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
	return `${(b / (1024 * 1024)).toFixed(2)} MB`;
};
var fmtDate = (iso) => {
	try {
		return new Date(iso).toLocaleDateString("en-US", {
			month: "short",
			day: "numeric",
			year: "numeric",
			hour: "2-digit",
			minute: "2-digit"
		});
	} catch {
		return iso;
	}
};
function ModelsPage() {
	const navigate = useNavigate();
	const [models, setModels] = (0, import_react.useState)([]);
	const [loading, setLoading] = (0, import_react.useState)(true);
	const [uploading, setUploading] = (0, import_react.useState)(false);
	const [uploadProgress, setUploadProgress] = (0, import_react.useState)(null);
	const [error, setError] = (0, import_react.useState)(null);
	const [deleting, setDeleting] = (0, import_react.useState)(null);
	const fileInputRef = (0, import_react.useRef)(null);
	const loadModels = (0, import_react.useCallback)(async () => {
		setLoading(true);
		setError(null);
		try {
			const data = await fetchModels();
			setModels(data);
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to load models");
		} finally {
			setLoading(false);
		}
	}, []);
	(0, import_react.useEffect)(() => {
		loadModels();
	}, [loadModels]);
	const handleUpload = async (e) => {
		const file = e.target.files?.[0];
		if (!file) return;
		setUploading(true);
		setUploadProgress(`Uploading ${file.name}...`);
		setError(null);
		try {
			const shapeStr = prompt("Enter input shape as comma-separated dimensions (e.g., 1,3,224,224):", "1,3,224,224");
			if (!shapeStr) {
				setUploading(false);
				setUploadProgress(null);
				return;
			}
			const inputShape = shapeStr.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => !Number.isNaN(n) && n > 0);
			if (inputShape.length === 0) throw new Error("Invalid shape. Enter numbers separated by commas.");
			setUploadProgress(`Validating and uploading ${file.name}...`);
			const result = await uploadModel(file, inputShape);
			setUploadProgress(`Uploaded ${result.name} successfully!`);
			await loadModels();
			setTimeout(() => setUploadProgress(null), 3e3);
		} catch (err) {
			setError(err instanceof Error ? err.message : "Upload failed");
			setUploadProgress(null);
		} finally {
			setUploading(false);
			if (fileInputRef.current) fileInputRef.current.value = "";
		}
	};
	const handleDelete = async (modelId, modelName) => {
		if (!confirm(`Delete model "${modelName}"? This cannot be undone.`)) return;
		setDeleting(modelId);
		try {
			await deleteModel(modelId);
			await loadModels();
		} catch (err) {
			setError(err instanceof Error ? err.message : "Delete failed");
		} finally {
			setDeleting(null);
		}
	};
	const handleUseForInference = (model) => {
		navigate({
			to: "/playground",
			search: {
				modelId: model.id,
				modelName: model.name
			}
		});
	};
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CrucibleLayout, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		className: "c-container",
		style: {
			paddingTop: 48,
			paddingBottom: 56
		},
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					display: "flex",
					justifyContent: "space-between",
					alignItems: "flex-start",
					flexWrap: "wrap",
					gap: 16,
					marginBottom: 32
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						className: "c-eyebrow",
						children: "MODEL MANAGEMENT"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
						className: "c-h2",
						style: { fontSize: 36 },
						children: "ONNX Models"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-muted",
						style: { marginTop: 8 },
						children: "Upload, manage, and run inference on your ONNX models."
					})
				] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					style: {
						display: "flex",
						gap: 8,
						alignItems: "center"
					},
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Button, {
							variant: "secondary",
							onClick: loadModels,
							disabled: loading,
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(RefreshCw, { size: 15 }), "Refresh"]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("input", {
							ref: fileInputRef,
							type: "file",
							accept: ".onnx",
							style: { display: "none" },
							onChange: handleUpload,
							disabled: uploading
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Button, {
							onClick: () => fileInputRef.current?.click(),
							disabled: uploading,
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Upload, { size: 15 }), "Upload Model"]
						})
					]
				})]
			}),
			uploadProgress && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					padding: "10px 14px",
					borderRadius: 2,
					background: "#ddeedf",
					border: "1px solid #b4d8be",
					color: "#166534",
					fontSize: 13,
					marginBottom: 20,
					display: "flex",
					alignItems: "center",
					gap: 8
				},
				children: [
					uploading && /* @__PURE__ */ (0, import_jsx_runtime.jsx)(RefreshCw, {
						size: 14,
						className: "c-spin"
					}),
					!uploading && /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Check, { size: 14 }),
					uploadProgress
				]
			}),
			error && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
				style: {
					padding: "10px 14px",
					borderRadius: 2,
					background: "#fadcdc",
					border: "1px solid #e4b4b4",
					color: "#b91c1c",
					fontSize: 13,
					marginBottom: 20
				},
				role: "alert",
				children: error
			}),
			loading ? /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					textAlign: "center",
					padding: 60
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(RefreshCw, {
					size: 24,
					className: "c-spin",
					style: { color: "var(--ink-muted)" }
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "c-muted",
					style: { marginTop: 12 },
					children: "Loading models..."
				})]
			}) : models.length === 0 ? /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Card, {
				style: {
					textAlign: "center",
					padding: 60,
					borderStyle: "dashed"
				},
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Inbox, {
						size: 40,
						style: {
							color: "var(--ink-subtle)",
							marginBottom: 16
						}
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
						className: "c-h3",
						style: {
							fontSize: 18,
							marginBottom: 8
						},
						children: "No models uploaded yet"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-muted",
						style: {
							maxWidth: 420,
							margin: "0 auto 20px"
						},
						children: "Upload an ONNX model file to start running inferences. Supported operators include MatMul, Relu, Sigmoid, Softmax, and more."
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						style: {
							display: "flex",
							gap: 10,
							justifyContent: "center",
							flexWrap: "wrap"
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Button, {
							onClick: () => fileInputRef.current?.click(),
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Upload, { size: 15 }), "Upload your first model"]
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
							to: "/playground",
							children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Button, {
								variant: "secondary",
								children: ["Try the playground ", /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ArrowRight, { size: 15 })]
							})
						})]
					})
				]
			}) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Card, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Table, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHeader, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(TableRow, { children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHead, { children: "Model Name" }),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHead, {
					style: { textAlign: "right" },
					children: "Size"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHead, {
					style: { textAlign: "right" },
					children: "Operators"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHead, {
					style: { textAlign: "right" },
					children: "Input Shape"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHead, {
					style: { textAlign: "right" },
					children: "Uploaded"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHead, {
					style: { textAlign: "right" },
					children: "Actions"
				})
			] }) }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableBody, { children: models.map((model) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(TableRow, { children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableCell, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					style: {
						display: "flex",
						alignItems: "center",
						gap: 8
					},
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)(FileUp, {
							size: 14,
							style: {
								color: "var(--trace)",
								flexShrink: 0
							}
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							style: { fontWeight: 500 },
							children: model.name
						}),
						!model.all_supported && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							style: {
								fontSize: 10,
								padding: "1px 6px",
								borderRadius: 2,
								background: "#f3e7c4",
								border: "1px solid #dcc58a",
								color: "#b45309",
								fontFamily: "var(--f-mono)"
							},
							children: "partial"
						})
					]
				}) }),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableCell, {
					style: { textAlign: "right" },
					children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
						style: {
							display: "inline-flex",
							alignItems: "center",
							gap: 4
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(HardDrive, {
							size: 12,
							style: { color: "var(--ink-muted)" }
						}), fmtBytes(model.file_size_bytes)]
					})
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableCell, {
					style: { textAlign: "right" },
					children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
						style: {
							display: "inline-flex",
							alignItems: "center",
							gap: 4
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Cpu, {
							size: 12,
							style: { color: "var(--ink-muted)" }
						}), model.operators.length]
					})
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(TableCell, {
					style: {
						textAlign: "right",
						fontFamily: "var(--f-mono)",
						fontSize: 12
					},
					children: [
						"[",
						model.input_shape?.join(", ") || "—",
						"]"
					]
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableCell, {
					style: { textAlign: "right" },
					children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
						style: {
							display: "inline-flex",
							alignItems: "center",
							gap: 4
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Clock, {
							size: 12,
							style: { color: "var(--ink-muted)" }
						}), fmtDate(model.created_at)]
					})
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableCell, {
					style: { textAlign: "right" },
					children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						style: {
							display: "flex",
							gap: 6,
							justifyContent: "flex-end"
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Button, {
							variant: "ghost",
							size: "sm",
							onClick: () => handleUseForInference(model),
							title: "Run inference",
							children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Play, { size: 14 })
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Button, {
							variant: "ghost",
							size: "sm",
							onClick: () => handleDelete(model.id, model.name),
							disabled: deleting === model.id,
							title: "Delete model",
							style: { color: "var(--risk)" },
							children: deleting === model.id ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)(RefreshCw, {
								size: 14,
								className: "c-spin"
							}) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Trash2, { size: 14 })
						})]
					})
				})
			] }, model.id)) })] }) })
		]
	}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("style", { children: `
        .c-spin {
          animation: c-spin 1s linear infinite;
        }
        @keyframes c-spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      ` })] });
}
var Route$8 = { component: ModelsPage };
var loginSchema = objectType({
	email: stringType().min(1, "Email is required").email("Please enter a valid email"),
	password: stringType().min(1, "Password is required")
});
function LoginPage() {
	const navigate = useNavigate();
	const [showPassword, setShowPassword] = (0, import_react.useState)(false);
	const [error, setError] = (0, import_react.useState)(null);
	const [loading, setLoading] = (0, import_react.useState)(false);
	const { register, handleSubmit, formState: { errors, isSubmitting } } = useForm({ resolver: u(loginSchema) });
	const onSubmit = async (data) => {
		setLoading(true);
		setError(null);
		try {
			await login(data.email, data.password);
			navigate({ to: "/dashboard" });
		} catch (err) {
			setError(err instanceof Error ? err.message : "Login failed. Please try again.");
		} finally {
			setLoading(false);
		}
	};
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CrucibleLayout, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
		className: "c-auth-page",
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Card, {
			className: "c-auth-card",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CardHeader, {
				className: "c-auth-header",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						className: "c-auth-icon",
						children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(LogIn, { size: 22 })
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CardTitle, {
						className: "c-auth-title",
						children: "Sign in"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CardDescription, {
						className: "c-auth-desc",
						children: "Access your dashboard and manage your inference models"
					})
				]
			}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CardContent, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("form", {
				onSubmit: handleSubmit(onSubmit),
				className: "c-auth-form",
				children: [
					error && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-auth-error",
						role: "alert",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CircleAlert, { size: 15 }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { children: error })]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-auth-field",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Label, {
								htmlFor: "email",
								children: "Email"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Input, {
								id: "email",
								type: "email",
								placeholder: "you@example.com",
								autoComplete: "email",
								...register("email"),
								className: errors.email ? "c-input-error" : ""
							}),
							errors.email && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "c-field-error",
								children: errors.email.message
							})
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-auth-field",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Label, {
								htmlFor: "password",
								children: "Password"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-password-wrap",
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Input, {
									id: "password",
									type: showPassword ? "text" : "password",
									placeholder: "Enter your password",
									autoComplete: "current-password",
									...register("password"),
									className: errors.password ? "c-input-error" : ""
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
									type: "button",
									className: "c-password-toggle",
									onClick: () => setShowPassword(!showPassword),
									"aria-label": showPassword ? "Hide password" : "Show password",
									children: showPassword ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)(EyeOff, { size: 15 }) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Eye, { size: 15 })
								})]
							}),
							errors.password && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "c-field-error",
								children: errors.password.message
							})
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Button, {
						type: "submit",
						className: "c-auth-submit",
						disabled: isSubmitting || loading,
						children: loading || isSubmitting ? /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(import_jsx_runtime.Fragment, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(LoaderCircle, {
							size: 16,
							className: "c-spin"
						}), "Signing in..."] }) : /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(import_jsx_runtime.Fragment, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(LogIn, { size: 16 }), "Sign in"] })
					})
				]
			}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
				className: "c-auth-footer",
				children: [
					"Don't have an account?",
					" ",
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
						to: "/register",
						className: "c-auth-link",
						children: "Create one"
					})
				]
			})] })]
		})
	}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("style", { children: `
        .c-auth-page {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 60vh;
          padding: 48px 20px;
        }
        .c-auth-card {
          width: 100%;
          max-width: 420px;
        }
        .c-auth-header {
          text-align: center;
          padding-bottom: 20px;
        }
        .c-auth-icon {
          width: 44px;
          height: 44px;
          border-radius: 50%;
          background: #fce8d5;
          border: 1px solid #ebc6a3;
          color: #c2410c;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 14px;
        }
        .c-auth-title {
          font-family: var(--f-serif);
          font-size: 28px;
          font-weight: 500;
          color: var(--ink);
          letter-spacing: -0.02em;
        }
        .c-auth-desc {
          color: var(--ink-muted);
          font-size: 14px;
          margin-top: 6px;
        }
        .c-auth-form {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .c-auth-field {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }
        .c-password-wrap {
          position: relative;
        }
        .c-password-wrap .c-input {
          padding-right: 40px;
        }
        .c-password-toggle {
          position: absolute;
          right: 10px;
          top: 50%;
          transform: translateY(-50%);
          background: none;
          border: none;
          cursor: pointer;
          color: var(--ink-muted);
          padding: 4px;
          display: flex;
          align-items: center;
        }
        .c-password-toggle:hover {
          color: var(--ink);
        }
        .c-input-error {
          border-color: var(--risk) !important;
        }
        .c-field-error {
          font-size: 12px;
          color: var(--risk);
          font-family: var(--f-mono);
        }
        .c-auth-error {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 12px;
          border-radius: 2px;
          background: #fadcdc;
          border: 1px solid #e4b4b4;
          color: var(--risk);
          font-size: 13px;
          line-height: 1.4;
        }
        .c-auth-submit {
          width: 100%;
          justify-content: center;
          margin-top: 4px;
        }
        .c-auth-footer {
          text-align: center;
          margin-top: 18px;
          font-size: 13px;
          color: var(--ink-muted);
        }
        .c-auth-link {
          color: var(--trace);
          font-weight: 600;
          text-decoration: underline;
          text-underline-offset: 2px;
        }
        .c-auth-link:hover {
          color: var(--forge-deep);
        }
        .c-spin {
          animation: c-spin 1s linear infinite;
        }
        @keyframes c-spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      ` })] });
}
var Route$7 = { component: LoginPage };
var $$splitComponentImporter$5 = () => import("./fraud-DtC3Yk1b.mjs");
var Route$6 = createFileRoute("/fraud")({
	head: () => ({ meta: [
		{ title: "Fraud Detector · Crucible" },
		{
			name: "description",
			content: "Run fraud inference on transactions entirely in-browser via Crucible's WASM ONNX runtime."
		},
		{
			property: "og:title",
			content: "Privacy-First Fraud Detection · Crucible"
		},
		{
			property: "og:description",
			content: "Zero network bytes. Zero data leaks. ML in the tab."
		}
	] }),
	component: lazyRouteComponent($$splitComponentImporter$5, "component")
});
var $$splitComponentImporter$4 = () => import("./docs-D0baYCHV.mjs");
var Route$5 = createFileRoute("/docs")({
	head: () => ({ meta: [
		{ title: "Operator Docs · Crucible" },
		{
			name: "description",
			content: "Reference documentation for every ONNX operator implemented in the Crucible engine."
		},
		{
			property: "og:title",
			content: "Crucible Operator Reference"
		},
		{
			property: "og:description",
			content: "Every kernel, attribute, and FFI signature."
		}
	] }),
	component: lazyRouteComponent($$splitComponentImporter$4, "component")
});
var $$splitComponentImporter$3 = () => import("./dashboard-DcTJB2Pq.mjs");
var Route$4 = createFileRoute("/dashboard")({
	head: () => ({ meta: [{ title: "Analytics Dashboard · Crucible" }, {
		name: "description",
		content: "Real-time analytics dashboard for Crucible inference platform."
	}] }),
	beforeLoad: () => {
		if (!localStorage.getItem("crucible_token")) throw redirect({ to: "/login" });
	},
	component: lazyRouteComponent($$splitComponentImporter$3, "component")
});
var $$splitComponentImporter$2 = () => import("./benchmark-Bw4k7LQe.mjs");
var Route$3 = createFileRoute("/benchmark")({
	head: () => ({ meta: [
		{ title: "Benchmark · Crucible" },
		{
			name: "description",
			content: "Head-to-head benchmarks: Crucible vs ONNX Runtime vs PyTorch across model sizes."
		},
		{
			property: "og:title",
			content: "Crucible Benchmark Console"
		},
		{
			property: "og:description",
			content: "Latency, footprint, cold-start — all measured, all in your browser."
		}
	] }),
	component: lazyRouteComponent($$splitComponentImporter$2, "component")
});
var $$splitComponentImporter$1 = () => import("./architecture-B_wPLN9o.mjs");
var Route$2 = createFileRoute("/architecture")({
	head: () => ({
		meta: [
			{ title: "Architecture · Crucible" },
			{
				name: "description",
				content: "How Crucible loads an ONNX model, decodes its protobuf bytes, and schedules kernels — both in native C++ and in Rust/WASM."
			},
			{
				property: "og:title",
				content: "Crucible — Architecture"
			},
			{
				property: "og:description",
				content: "Row-major NCHW tensors, Kahn's topological execution, and a hosted-vs-local data-path comparison."
			},
			{
				property: "og:url",
				content: "/architecture"
			}
		],
		links: [{
			rel: "canonical",
			href: "/architecture"
		}]
	}),
	component: lazyRouteComponent($$splitComponentImporter$1, "component")
});
var Dialog = Dialog$1;
var DialogPortal = DialogPortal$1;
var DialogOverlay = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)(DialogOverlay$1, {
	ref,
	className: cn("fixed inset-0 z-50 bg-black/80  data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0", className),
	...props
}));
DialogOverlay.displayName = DialogOverlay$1.displayName;
var DialogContent = import_react.forwardRef(({ className, children, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(DialogPortal, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(DialogOverlay, {}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(DialogContent$1, {
	ref,
	className: cn("fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 sm:rounded-lg", className),
	...props,
	children: [children, /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(DialogClose, {
		className: "absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background cursor-pointer transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground",
		children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(X, { className: "h-4 w-4" }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
			className: "sr-only",
			children: "Close"
		})]
	})]
})] }));
DialogContent.displayName = DialogContent$1.displayName;
var DialogHeader = ({ className, ...props }) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
	className: cn("flex flex-col space-y-1.5 text-center sm:text-left", className),
	...props
});
DialogHeader.displayName = "DialogHeader";
var DialogFooter = ({ className, ...props }) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
	className: cn("flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2", className),
	...props
});
DialogFooter.displayName = "DialogFooter";
var DialogTitle = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)(DialogTitle$1, {
	ref,
	className: cn("text-lg font-semibold leading-none tracking-tight", className),
	...props
}));
DialogTitle.displayName = DialogTitle$1.displayName;
var DialogDescription = import_react.forwardRef(({ className, ...props }, ref) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)(DialogDescription$1, {
	ref,
	className: cn("text-sm text-muted-foreground", className),
	...props
}));
DialogDescription.displayName = DialogDescription$1.displayName;
function ApiKeysPage() {
	const [keys, setKeys] = (0, import_react.useState)([]);
	const [loading, setLoading] = (0, import_react.useState)(true);
	const [error, setError] = (0, import_react.useState)(null);
	const [creating, setCreating] = (0, import_react.useState)(false);
	const [newKeyName, setNewKeyName] = (0, import_react.useState)("");
	const [newKeyExpiry, setNewKeyExpiry] = (0, import_react.useState)("");
	const [showCreateDialog, setShowCreateDialog] = (0, import_react.useState)(false);
	const [createdKey, setCreatedKey] = (0, import_react.useState)(null);
	const [copied, setCopied] = (0, import_react.useState)(false);
	const [revoking, setRevoking] = (0, import_react.useState)(null);
	const loadKeys = async () => {
		setLoading(true);
		setError(null);
		try {
			const data = await listApiKeys();
			setKeys(data);
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to load API keys");
		} finally {
			setLoading(false);
		}
	};
	(0, import_react.useEffect)(() => {
		loadKeys();
	}, []);
	const handleCreate = async () => {
		if (!newKeyName.trim()) return;
		setCreating(true);
		setError(null);
		try {
			const result = await createApiKey(newKeyName.trim(), newKeyExpiry ? parseInt(newKeyExpiry, 10) : void 0);
			setCreatedKey(result);
			await loadKeys();
			setNewKeyName("");
			setNewKeyExpiry("");
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to create API key");
		} finally {
			setCreating(false);
		}
	};
	const handleRevoke = async (keyId, keyName) => {
		if (!confirm(`Revoke API key "${keyName}"? This cannot be undone.`)) return;
		setRevoking(keyId);
		try {
			await revokeApiKey(keyId);
			await loadKeys();
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to revoke key");
		} finally {
			setRevoking(null);
		}
	};
	const copyKey = () => {
		if (createdKey?.key) {
			navigator.clipboard.writeText(createdKey.key);
			setCopied(true);
			setTimeout(() => setCopied(false), 2e3);
		}
	};
	const fmtDate = (iso) => {
		if (!iso) return "Never";
		try {
			return new Date(iso).toLocaleDateString("en-US", {
				month: "short",
				day: "numeric",
				year: "numeric"
			});
		} catch {
			return iso;
		}
	};
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(CrucibleLayout, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		className: "c-container",
		style: {
			paddingTop: 48,
			paddingBottom: 56
		},
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					display: "flex",
					justifyContent: "space-between",
					alignItems: "flex-start",
					flexWrap: "wrap",
					gap: 16,
					marginBottom: 32
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						className: "c-eyebrow",
						children: "API AUTHENTICATION"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
						className: "c-h2",
						style: { fontSize: 36 },
						children: "API Keys"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-muted",
						style: { marginTop: 8 },
						children: "Manage API keys for programmatic access to the inference engine."
					})
				] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Button, {
					onClick: () => setShowCreateDialog(true),
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Plus, { size: 15 }), "Create API Key"]
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Card, {
				style: {
					marginBottom: 24,
					borderStyle: "dashed"
				},
				children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CardContent, {
					style: {
						display: "flex",
						gap: 12,
						padding: 16
					},
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Info, {
						size: 18,
						style: {
							color: "var(--forge)",
							flexShrink: 0,
							marginTop: 2
						}
					}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
						className: "c-muted",
						style: { fontSize: 13 },
						children: [
							"Use API keys to authenticate programmatic requests. Include your key in the",
							" ",
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("code", {
								style: {
									fontFamily: "var(--f-mono)",
									fontSize: 12,
									background: "var(--paper-2)",
									padding: "1px 6px",
									borderRadius: 2
								},
								children: "X-API-Key"
							}),
							" ",
							"header. Keys are shown once at creation — save them securely. Never expose keys in client-side code."
						]
					}) })]
				})
			}),
			error && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
				style: {
					padding: "10px 14px",
					borderRadius: 2,
					background: "#fadcdc",
					border: "1px solid #e4b4b4",
					color: "#b91c1c",
					fontSize: 13,
					marginBottom: 20
				},
				role: "alert",
				children: error
			}),
			loading ? /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					textAlign: "center",
					padding: 60
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(RefreshCw, {
					size: 24,
					className: "c-spin",
					style: { color: "var(--ink-muted)" }
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "c-muted",
					style: { marginTop: 12 },
					children: "Loading API keys..."
				})]
			}) : keys.length === 0 ? /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Card, {
				style: {
					textAlign: "center",
					padding: 60,
					borderStyle: "dashed"
				},
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Key, {
						size: 40,
						style: {
							color: "var(--ink-subtle)",
							marginBottom: 16
						}
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
						className: "c-h3",
						style: {
							fontSize: 18,
							marginBottom: 8
						},
						children: "No API keys yet"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-muted",
						style: {
							maxWidth: 400,
							margin: "0 auto 20px"
						},
						children: "Create an API key to authenticate programmatic requests to the inference engine, model upload, and fraud detection endpoints."
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Button, {
						onClick: () => setShowCreateDialog(true),
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Plus, { size: 15 }), "Create your first API key"]
					})
				]
			}) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Card, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Table, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHeader, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(TableRow, { children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHead, { children: "Name" }),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHead, {
					style: { textAlign: "right" },
					children: "Prefix"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHead, {
					style: { textAlign: "right" },
					children: "Created"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHead, {
					style: { textAlign: "right" },
					children: "Expires"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHead, {
					style: { textAlign: "center" },
					children: "Status"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableHead, {
					style: { textAlign: "right" },
					children: "Actions"
				})
			] }) }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableBody, { children: keys.map((key) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(TableRow, { children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableCell, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					style: {
						display: "flex",
						alignItems: "center",
						gap: 8
					},
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Shield, {
						size: 14,
						style: { color: "var(--trace)" }
					}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
						style: { fontWeight: 500 },
						children: key.name
					})]
				}) }),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(TableCell, {
					style: {
						textAlign: "right",
						fontFamily: "var(--f-mono)",
						fontSize: 12,
						color: "var(--ink-muted)"
					},
					children: [key.prefix, "..."]
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableCell, {
					style: {
						textAlign: "right",
						fontSize: 13
					},
					children: fmtDate(key.created_at)
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableCell, {
					style: {
						textAlign: "right",
						fontSize: 13,
						color: key.expires_at ? "var(--ink-muted)" : "var(--ok)"
					},
					children: key.expires_at ? fmtDate(key.expires_at) : "Never"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableCell, {
					style: { textAlign: "center" },
					children: key.is_active ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
						style: {
							fontSize: 11,
							padding: "2px 8px",
							borderRadius: 2,
							background: "#ddeedf",
							border: "1px solid #b4d8be",
							color: "#166534",
							fontFamily: "var(--f-mono)"
						},
						children: "Active"
					}) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
						style: {
							fontSize: 11,
							padding: "2px 8px",
							borderRadius: 2,
							background: "#fadcdc",
							border: "1px solid #e4b4b4",
							color: "#b91c1c",
							fontFamily: "var(--f-mono)"
						},
						children: "Revoked"
					})
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TableCell, {
					style: { textAlign: "right" },
					children: key.is_active && /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Button, {
						variant: "ghost",
						size: "sm",
						onClick: () => handleRevoke(key.id, key.name),
						disabled: revoking === key.id,
						title: "Revoke key",
						style: { color: "var(--risk)" },
						children: revoking === key.id ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)(RefreshCw, {
							size: 14,
							className: "c-spin"
						}) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Trash2, { size: 14 })
					})
				})
			] }, key.id)) })] }) }),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Dialog, {
				open: showCreateDialog,
				onOpenChange: setShowCreateDialog,
				children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(DialogContent, { children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(DialogHeader, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(DialogTitle, { children: "Create API Key" }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(DialogDescription, { children: "Generate a new API key for programmatic access." })] }),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						style: {
							display: "flex",
							flexDirection: "column",
							gap: 14,
							marginTop: 12
						},
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Label, {
								htmlFor: "keyName",
								children: "Key name"
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Input, {
								id: "keyName",
								placeholder: "e.g., production-client",
								value: newKeyName,
								onChange: (e) => setNewKeyName(e.target.value),
								onKeyDown: (e) => e.key === "Enter" && handleCreate()
							})] }),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Label, {
								htmlFor: "keyExpiry",
								children: "Expiry (optional, days)"
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Input, {
								id: "keyExpiry",
								type: "number",
								min: "1",
								placeholder: "Leave empty for no expiry",
								value: newKeyExpiry,
								onChange: (e) => setNewKeyExpiry(e.target.value)
							})] }),
							error && !createdKey && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								style: {
									padding: "8px 12px",
									borderRadius: 2,
									background: "#fadcdc",
									border: "1px solid #e4b4b4",
									color: "#b91c1c",
									fontSize: 13
								},
								role: "alert",
								children: error
							})
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(DialogFooter, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Button, {
						variant: "secondary",
						onClick: () => setShowCreateDialog(false),
						children: "Cancel"
					}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Button, {
						onClick: handleCreate,
						disabled: !newKeyName.trim() || creating,
						children: creating ? /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(import_jsx_runtime.Fragment, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(RefreshCw, {
							size: 14,
							className: "c-spin"
						}), "Creating..."] }) : /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(import_jsx_runtime.Fragment, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Plus, { size: 14 }), "Create"] })
					})] })
				] })
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Dialog, {
				open: !!createdKey,
				onOpenChange: (open) => !open && setCreatedKey(null),
				children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(DialogContent, { children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(DialogHeader, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(DialogTitle, { children: "API Key Created" }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(DialogDescription, { children: "Save this key now — it will not be shown again." })] }),
					createdKey && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						style: { marginTop: 12 },
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: {
								padding: 12,
								borderRadius: 2,
								background: "var(--paper-2)",
								border: "1px solid var(--rule)",
								fontFamily: "var(--f-mono)",
								fontSize: 13,
								wordBreak: "break-all",
								color: "var(--ink)",
								display: "flex",
								alignItems: "center",
								gap: 8
							},
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("code", {
								style: {
									flex: 1,
									overflow: "hidden",
									textOverflow: "ellipsis"
								},
								children: createdKey.key
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
								onClick: copyKey,
								style: {
									background: "none",
									border: "1px solid var(--rule)",
									borderRadius: 2,
									padding: "4px 8px",
									cursor: "pointer",
									color: copied ? "var(--ok)" : "var(--ink-muted)",
									display: "flex",
									alignItems: "center",
									gap: 4,
									fontSize: 12,
									flexShrink: 0
								},
								children: [copied ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Check, { size: 14 }) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Copy, { size: 14 }), copied ? "Copied" : "Copy"]
							})]
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: {
								marginTop: 12,
								display: "flex",
								gap: 8,
								padding: "10px 12px",
								borderRadius: 2,
								background: "#f3e7c4",
								border: "1px solid #dcc58a",
								fontSize: 12,
								color: "#92400e",
								alignItems: "flex-start"
							},
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TriangleAlert, {
								size: 14,
								style: {
									flexShrink: 0,
									marginTop: 1
								}
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { children: "This key is only shown once. Copy it to a safe place before closing this dialog." })]
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(DialogFooter, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Button, {
						onClick: () => {
							setCreatedKey(null);
							setShowCreateDialog(false);
						},
						children: "Done"
					}) })
				] })
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("style", { children: `
          .c-spin {
            animation: c-spin 1s linear infinite;
          }
          @keyframes c-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        ` })
		]
	}) });
}
var Route$1 = { component: ApiKeysPage };
var $$splitComponentImporter = () => import("./routes-DRA8yZX2.mjs");
var Route = createFileRoute("/")({ component: lazyRouteComponent($$splitComponentImporter, "component") });
var StoryRoute = Route$12.update({
	id: "/story",
	path: "/story",
	getParentRoute: () => Route$13
});
var RoadmapRoute = Route$11.update({
	id: "/roadmap",
	path: "/roadmap",
	getParentRoute: () => Route$13
});
var RegisterRoute = Route$10.update({
	id: "/register",
	path: "/register",
	getParentRoute: () => Route$13
});
var PlaygroundRoute = Route$9.update({
	id: "/playground",
	path: "/playground",
	getParentRoute: () => Route$13
});
var ModelsRoute = Route$8.update({
	id: "/models",
	path: "/models",
	getParentRoute: () => Route$13
});
var LoginRoute = Route$7.update({
	id: "/login",
	path: "/login",
	getParentRoute: () => Route$13
});
var FraudRoute = Route$6.update({
	id: "/fraud",
	path: "/fraud",
	getParentRoute: () => Route$13
});
var DocsRoute = Route$5.update({
	id: "/docs",
	path: "/docs",
	getParentRoute: () => Route$13
});
var DashboardRoute = Route$4.update({
	id: "/dashboard",
	path: "/dashboard",
	getParentRoute: () => Route$13
});
var BenchmarkRoute = Route$3.update({
	id: "/benchmark",
	path: "/benchmark",
	getParentRoute: () => Route$13
});
var ArchitectureRoute = Route$2.update({
	id: "/architecture",
	path: "/architecture",
	getParentRoute: () => Route$13
});
var ApiKeysRoute = Route$1.update({
	id: "/api-keys",
	path: "/api-keys",
	getParentRoute: () => Route$13
});
var rootRouteChildren = {
	IndexRoute: Route.update({
		id: "/",
		path: "/",
		getParentRoute: () => Route$13
	}),
	ApiKeysRoute,
	ArchitectureRoute,
	BenchmarkRoute,
	DashboardRoute,
	DocsRoute,
	FraudRoute,
	LoginRoute,
	ModelsRoute,
	PlaygroundRoute,
	RegisterRoute,
	RoadmapRoute,
	StoryRoute
};
var routeTree = Route$13._addFileChildren(rootRouteChildren)._addFileTypes();
var getRouter = () => {
	return createRouter({
		routeTree,
		context: { queryClient: new QueryClient() },
		scrollRestoration: true,
		defaultPreloadStaleTime: 0
	});
};
//#endregion
export { getRouter };
