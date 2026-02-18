// import React, { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, AreaChart, Area, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Users, ShoppingCart, Package, Target, AlertTriangle, Lightbulb, ArrowUpRight, ArrowDownRight, Minus, Activity, Zap, RefreshCw, Download } from 'lucide-react';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import { useState, useEffect, useRef } from 'react';

// import"./index.css";

const API_BASE_URL = 'http://localhost:5010/api';

const ICON_MAP = {
  TrendingUp: TrendingUp,
  TrendingDown: TrendingDown,
  DollarSign: DollarSign,
  Users: Users,
  ShoppingCart: ShoppingCart,
  Package: Package,
  Target: Target,
  Activity: Activity
};

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

export default function KPIDashboard() {
  const [tables, setTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState('');
  const [kpiData, setKpiData] = useState(null);
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const dashboardRef = useRef(null);

  useEffect(() => {
    fetchTables();
  }, []);

  useEffect(() => {
    if (selectedTable) {
      loadKPIDashboard();
    }
  }, [selectedTable]);

  const fetchTables = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/tables`);
      const data = await response.json();
      if (data.success) {
        setTables(data.tables);
        if (data.tables.length > 0) {
          setSelectedTable(data.tables[0]);
        }
      }
    } catch (error) {
      console.error('Error fetching tables:', error);
    }
  };

  const loadKPIDashboard = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/dashboard/kpi/${selectedTable}`);
      const data = await response.json();
      if (data.success) {
        setKpiData(data);
        setLastUpdated(new Date());
        
        // Auto-generate insights
        await generateInsights(data.kpi_values, data.chart_data);
      }
    } catch (error) {
      console.error('Error loading KPI dashboard:', error);
    }
    setLoading(false);
  };

  const generateInsights = async (kpiValues, chartData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/dashboard/kpi/insights/${selectedTable}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kpi_values: kpiValues, chart_data: chartData })
      });
      const data = await response.json();
      if (data.success) {
        setInsights(data.insights);
      }
    } catch (error) {
      console.error('Error generating insights:', error);
    }
  };

  const handleDownloadPDF = async () => {
    if (!dashboardRef.current) return;
    
    setDownloading(true);
    try {
      const canvas = await html2canvas(dashboardRef.current, {
        scale: 2,
        useCORS: true,
        logging: false,
        windowWidth: dashboardRef.current.scrollWidth,
        windowHeight: dashboardRef.current.scrollHeight,
        onclone: (clonedDoc) => {
          // Fix gradient text for PDF capture
          const title = clonedDoc.querySelector('h1');
          if (title) {
            title.style.background = 'none';
            title.style.webkitTextFillColor = 'initial'; 
            title.style.color = '#60a5fa'; // solid blue color
          }

          // Fix Select Dropdown display - replace with text
          const select = clonedDoc.querySelector('select');
          if (select) {
            const textSpan = clonedDoc.createElement('span');
            textSpan.innerText = select.value;
            textSpan.style.color = '#f8fafc'; // slate-50
            textSpan.style.fontSize = '1rem';
            textSpan.style.fontWeight = '500';
            textSpan.style.padding = '0.5rem 1rem';
            textSpan.style.backgroundColor = '#334155'; // slate-700
            textSpan.style.border = '1px solid #475569'; // slate-600
            textSpan.style.borderRadius = '0.5rem';
            select.parentNode.replaceChild(textSpan, select);
          }
        }
      });
      
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      const imgWidth = canvas.width;
      const imgHeight = canvas.height;
      
      const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
      const imgX = (pdfWidth - imgWidth * ratio) / 2;
      const imgY = 10;
      
      // Calculate height based on width ratio to maintain aspect ratio
      const finalImgHeight = (imgHeight * pdfWidth) / imgWidth;
      
      // If content is longer than one page, we might need a different approach or multiple pages.
      // For now, let's fit to width and allow multi-page if needed (advanced) or just single page scaling.
      // Simple scaling to fit width:
      
      const imgHeightUpdated = (canvas.height * pdfWidth) / canvas.width;
      let heightLeft = imgHeightUpdated;
      let position = 0;

      pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, imgHeightUpdated);
      heightLeft -= pdfHeight;

      while (heightLeft >= 0) {
        position = heightLeft - imgHeightUpdated;
        pdf.addPage();
        pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, imgHeightUpdated);
        heightLeft -= pdfHeight;
      }
      
      pdf.save(`AI_Analytics_Report_${new Date().toISOString().split('T')[0]}.pdf`);
    } catch (error) {
      console.error('Error generating PDF:', error);
    }
    setDownloading(false);
  };

  const formatValue = (value, format) => {
    if (value === null || value === undefined) return 'N/A';
    
    switch (format) {
      case 'currency':
        return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      case 'percentage':
        return `${value.toFixed(2)}%`;
      case 'number':
        return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
      default:
        return value.toString();
    }
  };

  const getIcon = (iconName) => {
    const IconComponent = ICON_MAP[iconName] || Activity;
    return IconComponent;
  };

  const getTrendIcon = (direction) => {
    if (direction === 'up') return <ArrowUpRight className="text-green-400" size={20} />;
    if (direction === 'down') return <ArrowDownRight className="text-red-400" size={20} />;
    return <Minus className="text-slate-400" size={20} />;
  };

  const getCategoryColor = (category) => {
    const colors = {
      financial: 'from-green-500/10 to-emerald-600/10 border-green-500/20',
      operational: 'from-blue-500/10 to-blue-600/10 border-blue-500/20',
      customer: 'from-purple-500/10 to-purple-600/10 border-purple-500/20',
      product: 'from-orange-500/10 to-orange-600/10 border-orange-500/20'
    };
    return colors[category] || colors.operational;
  };

  const renderChart = (chartConfig) => {
    const { type, title, data, x_axis, y_axis } = chartConfig;

    if (!data || data.length === 0) return null;

    const chartProps = {
      data,
      margin: { top: 5, right: 30, left: 20, bottom: 5 }
    };

    switch (type) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart {...chartProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey={x_axis} stroke="#9ca3af" angle={-45} textAnchor="end" height={80} />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend />
              <Bar dataKey={y_axis} fill="#3b82f6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'line':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart {...chartProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey={x_axis} stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend />
              <Line type="monotone" dataKey={y_axis} stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6' }} />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'area':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart {...chartProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey={x_axis} stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend />
              <Area type="monotone" dataKey={y_axis} stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
            </AreaChart>
          </ResponsiveContainer>
        );

      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data}
                dataKey={y_axis}
                nameKey={x_axis}
                cx="50%"
                cy="50%"
                outerRadius={100}
                label
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  const getPriorityBadge = (priority) => {
    const styles = {
      high: 'bg-red-500/20 text-red-400 border-red-500/30',
      medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      low: 'bg-green-500/20 text-green-400 border-green-500/30'
    };
    return styles[priority] || styles.medium;
  };

  if (loading && !kpiData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="animate-spin mx-auto mb-4 text-blue-400" size={48} />
          <p className="text-xl">Analyzing your data and generating KPIs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6" ref={dashboardRef}>
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
                AI-Powered KPI Dashboard
              </h1>
              <p className="text-slate-400">Intelligent metrics generated from your data</p>
            </div>
            {lastUpdated && (
              <div className="text-sm text-slate-400">
                Last updated: {lastUpdated.toLocaleTimeString()}
              </div>
            )}
          </div>

          {/* Table Selector */}
          <div className="flex items-center gap-4 bg-slate-800/50 backdrop-blur-lg rounded-xl p-4 border border-slate-700/50">
            <label className="text-sm font-medium">Table:</label>
            <select
              value={selectedTable}
              onChange={(e) => setSelectedTable(e.target.value)}
              className="bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {tables.map(table => (
                <option key={table} value={table}>{table}</option>
              ))}
            </select>
            <div className="ml-auto flex gap-2" data-html2canvas-ignore="true">
              <button
                onClick={loadKPIDashboard}
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 px-4 py-2 rounded-lg flex items-center gap-2 transition-all"
              >
                <RefreshCw className={loading ? 'animate-spin' : ''} size={18} />
                Refresh
              </button>
              <button
                 onClick={handleDownloadPDF}
                 disabled={downloading || loading}
                 className="bg-purple-600 hover:bg-purple-700 disabled:opacity-50 px-4 py-2 rounded-lg flex items-center gap-2 transition-all text-white font-medium shadow-lg shadow-purple-900/20"
               >
                 <Download className={downloading ? 'animate-bounce' : ''} size={18} />
                 {downloading ? 'Downloading...' : 'Download Report'}
               </button>
            </div>
          </div>
        </div>

        {kpiData && (
          <div className="space-y-6">
            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {kpiData.kpi_values?.map((kpi, idx) => {
                const IconComponent = getIcon(kpi.icon);
                return (
                  <div
                    key={idx}
                    className={`bg-gradient-to-br ${getCategoryColor(kpi.category)} backdrop-blur-lg rounded-xl p-6 border transition-all hover:scale-105`}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="bg-slate-700/50 p-3 rounded-lg">
                        <IconComponent size={24} className="text-blue-400" />
                      </div>
                      {getTrendIcon(kpi.change_direction)}
                    </div>
                    <h3 className="text-sm font-medium text-slate-400 mb-1">{kpi.name}</h3>
                    <p className="text-3xl font-bold mb-1">{formatValue(kpi.value, kpi.format)}</p>
                    <p className="text-xs text-slate-500">{kpi.description}</p>
                  </div>
                );
              })}
            </div>

            {/* Charts */}
            {kpiData.chart_data && kpiData.chart_data.length > 0 && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {kpiData.chart_data.map((chart, idx) => (
                  <div key={idx} className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-slate-700/50">
                    <div className="mb-4">
                      <h3 className="text-xl font-semibold mb-1">{chart.title}</h3>
                      <p className="text-sm text-slate-400">{chart.description}</p>
                    </div>
                    {renderChart(chart)}
                  </div>
                ))}
              </div>
            )}

            {/* AI Insights */}
            {insights && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Key Observations */}
                <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-slate-700/50">
                  <div className="flex items-center gap-2 mb-4">
                    <Activity className="text-blue-400" size={24} />
                    <h3 className="text-xl font-semibold">Key Observations</h3>
                  </div>
                  <div className="space-y-2">
                    {insights.observations?.map((obs, idx) => (
                      <div key={idx} className="flex items-start gap-3 bg-slate-700/30 rounded-lg p-3">
                        <div className="bg-blue-500/20 rounded-full p-1 mt-0.5">
                          <div className="w-2 h-2 bg-blue-400 rounded-full" />
                        </div>
                        <p className="text-sm text-slate-300">{obs}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Action Items */}
                <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-slate-700/50">
                  <div className="flex items-center gap-2 mb-4">
                    <Target className="text-purple-400" size={24} />
                    <h3 className="text-xl font-semibold">Action Items</h3>
                  </div>
                  <div className="space-y-3">
                    {insights.action_items?.map((item, idx) => (
                      <div key={idx} className="bg-slate-700/30 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-1">
                          <h4 className="font-medium text-sm">{item.title}</h4>
                          <span className={`text-xs px-2 py-1 rounded-full border ${getPriorityBadge(item.priority)}`}>
                            {item.priority}
                          </span>
                        </div>
                        <p className="text-sm text-slate-400">{item.description}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Opportunities */}
                <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-slate-700/50">
                  <div className="flex items-center gap-2 mb-4">
                    <Lightbulb className="text-yellow-400" size={24} />
                    <h3 className="text-xl font-semibold">Opportunities</h3>
                  </div>
                  <div className="space-y-2">
                    {insights.opportunities?.map((opp, idx) => (
                      <div key={idx} className="flex items-start gap-3 bg-gradient-to-r from-yellow-500/10 to-orange-500/10 border border-yellow-500/20 rounded-lg p-3">
                        <Zap className="text-yellow-400 flex-shrink-0 mt-0.5" size={18} />
                        <p className="text-sm text-slate-300">{opp}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Risks */}
                <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-slate-700/50">
                  <div className="flex items-center gap-2 mb-4">
                    <AlertTriangle className="text-red-400" size={24} />
                    <h3 className="text-xl font-semibold">Risks & Concerns</h3>
                  </div>
                  <div className="space-y-2">
                    {insights.risks?.map((risk, idx) => (
                      <div key={idx} className="flex items-start gap-3 bg-gradient-to-r from-red-500/10 to-orange-500/10 border border-red-500/20 rounded-lg p-3">
                        <AlertTriangle className="text-red-400 flex-shrink-0 mt-0.5" size={18} />
                        <p className="text-sm text-slate-300">{risk}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Data Summary */}
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-slate-700/50">
              <h3 className="text-xl font-semibold mb-4">Data Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <p className="text-3xl font-bold text-blue-400">{kpiData.total_records.toLocaleString()}</p>
                  <p className="text-sm text-slate-400 mt-1">Total Records</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-green-400">
                    {kpiData.data_profile?.columns ? Object.keys(kpiData.data_profile.columns).length : 0}
                  </p>
                  <p className="text-sm text-slate-400 mt-1">Columns</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-purple-400">{kpiData.kpi_values?.length || 0}</p>
                  <p className="text-sm text-slate-400 mt-1">Active KPIs</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-orange-400">{kpiData.chart_data?.length || 0}</p>
                  <p className="text-sm text-slate-400 mt-1">Visualizations</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}