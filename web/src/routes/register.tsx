import { useState } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Eye, EyeOff, Loader2, UserPlus, AlertCircle, Check } from "lucide-react";
import { register as registerAccount } from "../lib/api";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { CrucibleLayout } from "../components/crucible/Layout";

const registerSchema = z
  .object({
    fullName: z.string().min(1, "Full name is required"),
    email: z.string().min(1, "Email is required").email("Please enter a valid email"),
    password: z.string().min(8, "Password must be at least 8 characters"),
    confirmPassword: z.string().min(1, "Please confirm your password"),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords do not match",
    path: ["confirmPassword"],
  });

type RegisterForm = z.infer<typeof registerSchema>;

function RegisterPage() {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<RegisterForm>({
    resolver: zodResolver(registerSchema),
  });

  const onSubmit = async (data: RegisterForm) => {
    setLoading(true);
    setError(null);
    try {
      await registerAccount(data.email, data.password, data.fullName);
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

  if (success) {
    return (
      <CrucibleLayout>
        <div className="c-auth-page">
          <Card className="c-auth-card">
            <CardContent style={{ textAlign: "center", padding: "40px 24px" }}>
              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: "50%",
                  background: "#ddeedf",
                  border: "1px solid #b4d8be",
                  color: "#166534",
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginBottom: 16,
                }}
              >
                <Check size={24} />
              </div>
              <h2
                style={{
                  fontFamily: "var(--f-serif)",
                  fontSize: 24,
                  fontWeight: 500,
                  color: "var(--ink)",
                  marginBottom: 8,
                }}
              >
                Account created
              </h2>
              <p style={{ color: "var(--ink-muted)", fontSize: 14 }}>
                Redirecting to your dashboard...
              </p>
            </CardContent>
          </Card>
        </div>
        <style>{`
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
        `}</style>
      </CrucibleLayout>
    );
  }

  return (
    <CrucibleLayout>
      <div className="c-auth-page">
        <Card className="c-auth-card">
          <CardHeader className="c-auth-header">
            <div className="c-auth-icon">
              <UserPlus size={22} />
            </div>
            <CardTitle className="c-auth-title">Create account</CardTitle>
            <CardDescription className="c-auth-desc">
              Start running ONNX inference and fraud detection
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="c-auth-form">
              {error && (
                <div className="c-auth-error" role="alert">
                  <AlertCircle size={15} />
                  <span>{error}</span>
                </div>
              )}

              <div className="c-auth-field">
                <Label htmlFor="fullName">Full name</Label>
                <Input
                  id="fullName"
                  type="text"
                  placeholder="Your name"
                  autoComplete="name"
                  {...register("fullName")}
                  className={errors.fullName ? "c-input-error" : ""}
                />
                {errors.fullName && (
                  <span className="c-field-error">{errors.fullName.message}</span>
                )}
              </div>

              <div className="c-auth-field">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  autoComplete="email"
                  {...register("email")}
                  className={errors.email ? "c-input-error" : ""}
                />
                {errors.email && <span className="c-field-error">{errors.email.message}</span>}
              </div>

              <div className="c-auth-field">
                <Label htmlFor="password">Password</Label>
                <div className="c-password-wrap">
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="At least 8 characters"
                    autoComplete="new-password"
                    {...register("password")}
                    className={errors.password ? "c-input-error" : ""}
                  />
                  <button
                    type="button"
                    className="c-password-toggle"
                    onClick={() => setShowPassword(!showPassword)}
                    aria-label={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
                {errors.password && (
                  <span className="c-field-error">{errors.password.message}</span>
                )}
              </div>

              <div className="c-auth-field">
                <Label htmlFor="confirmPassword">Confirm password</Label>
                <div className="c-password-wrap">
                  <Input
                    id="confirmPassword"
                    type={showConfirm ? "text" : "password"}
                    placeholder="Repeat your password"
                    autoComplete="new-password"
                    {...register("confirmPassword")}
                    className={errors.confirmPassword ? "c-input-error" : ""}
                  />
                  <button
                    type="button"
                    className="c-password-toggle"
                    onClick={() => setShowConfirm(!showConfirm)}
                    aria-label={showConfirm ? "Hide password" : "Show password"}
                  >
                    {showConfirm ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
                {errors.confirmPassword && (
                  <span className="c-field-error">{errors.confirmPassword.message}</span>
                )}
              </div>

              <Button type="submit" className="c-auth-submit" disabled={isSubmitting || loading}>
                {loading || isSubmitting ? (
                  <>
                    <Loader2 size={16} className="c-spin" />
                    Creating account...
                  </>
                ) : (
                  <>
                    <UserPlus size={16} />
                    Create account
                  </>
                )}
              </Button>
            </form>

            <p className="c-auth-footer">
              Already have an account?{" "}
              <Link to="/login" className="c-auth-link">
                Sign in
              </Link>
            </p>
          </CardContent>
        </Card>
      </div>

      <style>{`
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
      `}</style>
    </CrucibleLayout>
  );
}

export const Route = {
  component: RegisterPage,
};
