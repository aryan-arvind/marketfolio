import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Trading Terminal ErrorBoundary caught:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#0c1017] text-[#e5e7eb] flex items-center justify-center p-6">
          <div className="max-w-lg w-full bg-[#131722] border border-[#f23645]/40 rounded-lg p-6 shadow-2xl">
            <div className="flex items-center gap-3 mb-4 text-[#f23645]">
              <AlertTriangle size={24} />
              <h2 className="text-lg font-bold font-mono">Terminal Rendering Interrupted</h2>
            </div>
            <p className="text-xs text-[#94a3b8] mb-4 leading-relaxed font-sans">
              An unexpected presentation error occurred while rendering the instrument workspace. Your underlying analytical engine and Python kernel remain unaffected.
            </p>
            <div className="bg-[#0c1017] border border-[#1e222d] p-3 rounded text-[11px] font-mono text-[#f87171] mb-5 overflow-x-auto max-h-36">
              {this.state.error?.message || 'Unknown runtime error'}
            </div>
            <button
              onClick={this.handleReset}
              className="w-full flex items-center justify-center gap-2 bg-[#06b6d4] hover:bg-[#0891b2] text-[#0c1017] font-semibold text-xs py-2.5 rounded transition-all shadow-md font-sans"
            >
              <RefreshCw size={14} />
              <span>Reset & Reload Trading Desk</span>
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
